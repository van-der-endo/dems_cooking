import xarray as xr
import numpy as np
from typing import List

#############################################
# Functions which are used for PSW data
#############################################

def remove_bad_channels(da, bad_indices):
    """
    Removes data from the DataArray where the 'chan' coordinate matches values in bad_indices.

    Parameters:
    da (xarray.DataArray): Input DataArray with a 'chan' coordinate.
    bad_indices (numpy.ndarray): Array of integers representing the indices of bad channels.

    Returns:
    xarray.DataArray: The DataArray with bad channels removed.
    """
    # Ensure bad_indices is a numpy array
    bad_indices = np.array(bad_indices)
    
    # Mask the DataArray to exclude bad channels
    da_cleaned = da.where(~da['chan'].isin(bad_indices), drop=True)
    
    return da_cleaned


def filter_on_off_states(da: xr.DataArray) -> xr.DataArray:
    """
    Filter the DataArray to keep only the 'ON' and 'OFF' states along the 'time' dimension.

    Parameters:
    - da (xr.DataArray): The input DataArray with a 'state' coordinate along 'time'.

    Returns:
    - xr.DataArray: A new DataArray containing only 'ON' and 'OFF' states.
    """
    # Select indices where 'state' is either 'ON' or 'OFF'
    valid_states = da['state'].isin(['ON', 'OFF'])
    
    # Filter the DataArray
    filtered_da = da.sel(time=valid_states)
    
    return filtered_da


def remove_first_last_in_chops(da):
    """
    Removes the first and last indices of every consecutive occurrence of "A" or "B" in the 'beam' coordinate.
    
    Parameters:
    - da: xarray.DataArray
        Input DataArray with a 'beam' coordinate.
        
    Returns:
    - xarray.DataArray
        Modified DataArray with selected indices removed.
    """
    beam_values = da.beam.values
    to_remove = []

    # Iterate through beam values to find sequences of "A" and "B"
    start_idx = None
    current_value = None

    for i, value in enumerate(beam_values):
        if value in {"A", "B"}:
            if value != current_value:
                # New group starts
                if start_idx is not None:
                    # End of previous group
                    to_remove.append(start_idx)  # First index
                    to_remove.append(i - 1)  # Last index
                start_idx = i
                current_value = value
        else:
            # Not "A" or "B", end the current group
            if start_idx is not None:
                to_remove.append(start_idx)  # First index
                to_remove.append(i - 1)  # Last index
                start_idx = None
                current_value = None

    # Handle last group if it reaches the end of the array
    if start_idx is not None:
        to_remove.append(start_idx)  # First index
        to_remove.append(len(beam_values) - 1)  # Last index

    # Create a mask of indices to keep
    to_keep = np.ones(da.time.size, dtype=bool)
    to_keep[to_remove] = False

    # Return the filtered DataArray
    return da.isel(time=to_keep)

def trim_by_nod(da: xr.DataArray) -> xr.DataArray:
    """
    Trims the DataArray along the 'time' index such that the 'nod' coordinate
    starts at the first occurrence of 'A2' and ends at the last occurrence of 'A1'.

    Parameters:
    - da (xr.DataArray): The input DataArray with 'time' and 'nod' coordinates.

    Returns:
    - xr.DataArray: The trimmed DataArray.
    """
    # Get the indices of the first 'A2' and the last 'A1' in the 'nod' coordinate
    first_a2_idx = (da.nod == 'A2').argmax().item()
    last_a1_idx = len(da.nod) - 1 - (da.nod[::-1] == 'A1').argmax().item()
    
    # Trim the DataArray using these indices
    return da.isel(time=slice(first_a2_idx, last_a1_idx + 1))


def trim_dataarray_by_beam_and_nod(da):
    """
    Trim an Xarray DataArray based on 'beam' and 'nod' coordinates within each block.

    The function performs the following steps:
    1. Divides the DataArray into blocks based on consecutive values in the 'nod' coordinate.
    2. Within each 'nod' block, identifies chunks of consecutive 'A' and 'B' values in the 'beam' coordinate.
    3. Ensures each trimmed block starts with the second 'A' chunk and ends with the last 'B' chunk.
    4. Removes the first (A, B) chunk set if the first 'A' chunk has 3 points or fewer.
    5. Removes the last (A, B) chunk set if the last 'B' chunk has 3 points or fewer.
    6. Combines the trimmed blocks into a single DataArray.

    Parameters:
    -----------
    da : xarray.DataArray
        The input DataArray with the following properties:
        - Dimensions: 'time' and 'chan'.
        - Coordinates:
          - 'beam' : Contains 'A' or 'B' values along the 'time' dimension.
          - 'nod'  : Contains nod status strings ('A2', 'B1', 'B2', 'A1') along the 'time' dimension.
          - 'ABBA' : Contains cycle numbers along the 'time' dimension.

    Returns:
    --------
    xarray.DataArray
        A trimmed DataArray where each block starts with 'A' and ends with 'B', with insufficient edge chunks removed.
    """

    def find_chunks(values):
        """Identify chunks of consecutive identical values and their start and end indices."""
        chunks = []
        start = 0
        for i in range(1, len(values)):
            if values[i] != values[start]:
                chunks.append((values[start], start, i))
                start = i
        # Add the last chunk
        chunks.append((values[start], start, len(values)))
        return chunks

    # Initialize an empty list to store trimmed slices
    trimmed_slices = []

    # Get unique nod values to identify blocks
    nod_values = da.nod.values
    nod_chunks = find_chunks(nod_values)

    for _, start, end in nod_chunks:
        # Select the current nod block
        block = da.isel(time=slice(start, end))
        beam_values = block.beam.values

        # Find the chunks of consecutive 'beam' values within this block
        beam_chunks = find_chunks(beam_values)

        # Ensure there are at least two 'A' chunks and at least one 'B' chunk
        a_chunks = [(value, start, end) for value, start, end in beam_chunks if value == 'A']
        b_chunks = [(value, start, end) for value, start, end in beam_chunks if value == 'B']

        if len(a_chunks) >= 2 and len(b_chunks) >= 1:
            # Check the first A chunk and the last B chunk
            first_a_chunk = a_chunks[0]
            last_b_chunk = b_chunks[-1]

            # Check if the first A chunk has 3 points or fewer
            if (first_a_chunk[2] - first_a_chunk[1]) <= 3:
                # Remove the first (A, B) chunk set if possible
                if len(a_chunks) >= 3 and len(b_chunks) >= 2:
                    a_chunks.pop(0)
                    b_chunks.pop(0)

            # Check if the last B chunk has 3 points or fewer
            if (last_b_chunk[2] - last_b_chunk[1]) <= 3:
                # Remove the last (A, B) chunk set if possible
                if len(a_chunks) >= 3 and len(b_chunks) >= 2:
                    a_chunks.pop(-1)
                    b_chunks.pop(-1)

            # Identify indices for the second 'A' chunk and the last 'B' chunk
            second_a_index = a_chunks[1][1]
            last_b_index = b_chunks[-1][2]

            # Trim the block to the desired range
            trimmed_block = block.isel(time=slice(second_a_index, last_b_index))
            trimmed_slices.append(trimmed_block)

    # Combine all trimmed slices into a single DataArray
    if trimmed_slices:
        return xr.concat(trimmed_slices, dim='time')
    else:
        return xr.DataArray()

def divide_da_by_ABBA(da_abba: xr.DataArray) -> List[xr.DataArray]:
    """
    Divide da_abba into a list of DataArrays based on unique values in the `ABBA` coordinate.

    Args:
        da_abba: The input DataArray with `ABBA` as a coordinate.

    Returns:
        A list of DataArrays, one for each unique value in the `ABBA` coordinate.
    """
    if "ABBA" not in da_abba.coords:
        raise ValueError("The input DataArray does not have an `ABBA` coordinate.")
    
    # Get the unique ABBA values
    unique_abbas = da_abba["ABBA"].values
    unique_groups = np.unique(unique_abbas)

    # Split the DataArray by ABBA groups
    da_list = [da_abba.sel(time=da_abba["ABBA"] == group) for group in unique_groups]

    return da_list

#############################################
# Functions which are used for Still data
#############################################

def trim_to_scan(da):
    """
    Trims the DataArray `da` to only include elements between the first and last occurrence of "SCAN" in da.state.values.

    Parameters:
        da (xarray.DataArray): Input DataArray with a 'state' coordinate containing strings.

    Returns:
        xarray.DataArray: Trimmed DataArray with elements before the first "SCAN" and after the last "SCAN" removed.
    """
    # Extract the 'state' coordinate values
    state_values = da.state.values

    # Find indices of "SCAN"
    scan_indices = np.where(state_values == "SCAN")[0]

    # If "SCAN" is not found, return an empty DataArray
    if scan_indices.size == 0:
        return da.isel(time=slice(0, 0))

    # Get the first and last indices of "SCAN"
    first_scan_index = scan_indices[0]
    last_scan_index = scan_indices[-1]

    # Trim the DataArray
    da_trimmed = da.isel(time=slice(first_scan_index, last_scan_index + 1))

    return da_trimmed

def trim_da_to_complete_ab_beam_pairs(da):
    """
    Trims the DataArray based on two conditions:
      1. Removes all data before the first index of the second consecutive "A".
      2. Removes all data after the last index of the second-to-last consecutive "B".
    
    Parameters:
        da (xarray.DataArray): The input DataArray with a 'beam' coordinate.
    
    Returns:
        xarray.DataArray: The trimmed DataArray.
    """
    # Helper to find the first index of the second consecutive "A"
    def find_first_index_second_consecutive_a(beam_values):
        count_consecutive = 0
        for i in range(1, len(beam_values)):
            if beam_values[i] == "A" and beam_values[i - 1] == "A":
                count_consecutive += 1
                if count_consecutive == 2:
                    return i - 1
        return None

    # Helper to find the last index of the second-to-last consecutive "B"
    def find_last_index_second_last_consecutive_b(beam_values):
        consecutive_indices = []
        for i in range(1, len(beam_values)):
            if beam_values[i] == "B" and beam_values[i - 1] == "B":
                consecutive_indices.append(i - 1)
        if len(consecutive_indices) < 2:
            return None
        return consecutive_indices[-2]

    # Extract beam values
    beam_values = da.beam.values

    # Find the trimming indices
    start_index = find_first_index_second_consecutive_a(beam_values)
    end_index = find_last_index_second_last_consecutive_b(beam_values)

    # Handle the slicing
    if start_index is None and end_index is None:
        return da  # No trimming needed
    elif start_index is None:
        return da.isel(time=slice(None, end_index + 2))  # Only trim after
    elif end_index is None:
        return da.isel(time=slice(start_index, None))  # Only trim before
    else:
        return da.isel(time=slice(start_index, end_index + 2))  # Trim both sides
    
