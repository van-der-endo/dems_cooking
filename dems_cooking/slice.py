import xarray as xr
import numpy as np
from typing import List

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
    

    import xarray as xr
import numpy as np

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