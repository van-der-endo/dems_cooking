import xarray as xr
import numpy as np
import pandas as pd

def add_ABBA_coordinate(da, n_chops_in_ABBA_cycle):
    # Validation checks
    if not isinstance(n_chops_in_ABBA_cycle, int) or n_chops_in_ABBA_cycle % 4 != 0:
        raise ValueError("n_chops_in_ABBA_cycle must be an integer and a multiple of 4.")
    
    if da.beam.values[0] != 'A':
        raise ValueError("The first value in the 'beam' coordinate must be 'A'.")
    
    if da.beam.values[-1] != 'B':
        raise ValueError("The last value in the 'beam' coordinate must be 'B'.")

    # Identify indices of chop-pairs
    beam = da.beam.values
    chop_pairs = []
    i = 0
    while i < len(beam) - 1:
        if beam[i] == 'A':
            j = i
            while j < len(beam) and beam[j] == 'A':  # Find end of consecutive A
                j += 1
            if j < len(beam) and beam[j] == 'B':  # Start of consecutive B
                k = j
                while k < len(beam) and beam[k] == 'B':  # Find end of consecutive B
                    k += 1
                chop_pairs.append((i, k))  # Store chop-pair range
                i = k  # Move to next potential chop-pair
            else:
                break
        else:
            i += 1

    # Assign ABBA cycles
    ABBA = np.full(len(beam), np.nan)  # Initialize with NaN
    cycle_index = 0
    valid_chop_pairs = len(chop_pairs) // n_chops_in_ABBA_cycle * n_chops_in_ABBA_cycle  # Trim incomplete cycles
    for cycle_start in range(0, valid_chop_pairs, n_chops_in_ABBA_cycle):
        for i in range(cycle_start, cycle_start + n_chops_in_ABBA_cycle):
            start, end = chop_pairs[i]
            ABBA[start:end] = cycle_index  # Assign ABBA cycle
        cycle_index += 1

    # Trim data based on valid chop-pairs
    if valid_chop_pairs < len(chop_pairs):
        trim_end = chop_pairs[valid_chop_pairs - 1][1]  # Last valid index
        da = da.sel(time=slice(da.time.values[0], da.time.values[trim_end - 1]))

    # Add ABBA as a coordinate
    da = da.assign_coords(ABBA=("time", ABBA[:len(da.time)]))

    return da


def add_nod_coordinate(da_abba, n_chops_in_ABBA_cycle):
    # Validation check
    if not isinstance(n_chops_in_ABBA_cycle, int) or n_chops_in_ABBA_cycle % 4 != 0:
        raise ValueError("n_chops_in_ABBA_cycle must be an integer and a multiple of 4.")
    
    # Initialize the new 'nod' coordinate
    nod = np.full(len(da_abba.time), "", dtype=object)
    ABBA = da_abba.ABBA.values  # Retrieve the ABBA coordinate
    beam = da_abba.beam.values

    # Define nod assignment order
    nod_labels = ["A2", "B1", "B2", "A1"]
    chop_size = n_chops_in_ABBA_cycle // 4  # Number of chop-pairs per nod segment

    # Process each ABBA group separately
    for abba_group in np.unique(ABBA[~np.isnan(ABBA)]):  # Iterate through ABBA groups
        indices = np.where(ABBA == abba_group)[0]  # Find indices for this ABBA group
        
        # Identify chop-pairs within this ABBA group
        chop_pairs = []
        i = indices[0]
        while i < indices[-1]:
            if beam[i] == "A":
                j = i
                while j < len(beam) and j <= indices[-1] and beam[j] == "A":
                    j += 1
                if j < len(beam) and j <= indices[-1] and beam[j] == "B":
                    k = j
                    while k < len(beam) and k <= indices[-1] and beam[k] == "B":
                        k += 1
                    chop_pairs.append((i, k))  # Add chop-pair
                    i = k
                else:
                    break
            else:
                i += 1

        # Assign 'nod' labels for chop-pairs within the ABBA group
        for i, label in enumerate(nod_labels):
            start_idx = i * chop_size
            end_idx = (i + 1) * chop_size
            for start, end in chop_pairs[start_idx:end_idx]:
                nod[start:end] = label

    # Add 'nod' as a new coordinate
    da_abba = da_abba.assign_coords(nod=("time", nod))
    
    return da_abba

def add_nod_coordinate_based_on_ON_OFF(da):
    # Ensure "state" coordinate exists along "time"
    if "state" not in da.coords or "time" not in da.dims:
        raise ValueError("The DataArray must have a 'state' coordinate along the 'time' dimension.")
    
    # Get the states and their values
    states = da.coords['state'].values
    times = da['time'].values
    
    # Identify consecutive segments where state is the same
    state_groups = pd.Series(states, index=times).groupby((states != np.roll(states, 1)).cumsum())
    
    # Initialize an empty array to store the nod labels
    nod_labels = np.empty(len(times), dtype=object)
    
    # Iterate through each consecutive state group
    for _, group in state_groups:
        state_value = group.iloc[0]  # Either 'ON' or 'OFF'
        indices = group.index
        
        # Skip if the segment is empty
        if len(indices) == 0:
            continue
        
        # If the number of points is odd, drop the last point
        if len(indices) % 2 != 0:
            indices = indices[:-1]
        
        # Split the indices into two halves
        half_len = len(indices) // 2
        first_half = indices[:half_len]
        second_half = indices[half_len:]
        
        # Assign appropriate nod labels
        if state_value == "ON":
            nod_labels[da.get_index('time').get_indexer(first_half)] = "A1"
            nod_labels[da.get_index('time').get_indexer(second_half)] = "A2"
        elif state_value == "OFF":
            nod_labels[da.get_index('time').get_indexer(first_half)] = "B1"
            nod_labels[da.get_index('time').get_indexer(second_half)] = "B2"
    
    # Filter out indices where nod_labels are empty (i.e., '')
    valid_indices = nod_labels != ''
    da_filtered = da.isel(time=valid_indices)
    
    # Only keep the valid 'nod' labels
    da_filtered = da_filtered.assign_coords(nod=("time", nod_labels[valid_indices]))
    
    return da_filtered





def add_ABBA_cycle_coordinate(da):
    # Initialize an empty list to store the ABBA block labels
    abba_labels = []

    # Keep track of the current block counters for each nod value
    block_counters = {'A2': -1, 'B1': -1, 'B2': -1, 'A1': -1}
    current_nod = None

    # Iterate over the values in 'nod' to create the ABBA blocks
    for nod_value in da.nod.values:
        # Skip empty or invalid nod values
        if nod_value not in block_counters:
            abba_labels.append(np.nan)  # You could append NaN for invalid values
            continue
        
        # Check if we're entering a new block
        if nod_value != current_nod:
            # If the value changed, reset the block counter for that value
            block_counters[nod_value] += 1
            current_nod = nod_value
        
        # Add the current block label for the specific nod value
        abba_labels.append(block_counters[nod_value])

    # Convert the list of labels into a numpy array
    abba_labels = np.array(abba_labels)

    # Add the new coordinate 'ABBA' to the DataArray
    da = da.assign_coords(ABBA=('time', abba_labels))

    return da