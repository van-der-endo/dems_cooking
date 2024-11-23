import xarray as xr
import numpy as np

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