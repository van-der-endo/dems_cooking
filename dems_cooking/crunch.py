import numpy as np
import xarray as xr

def calculate_chop_means(da):
    """
    Calculate the means of consecutive occurrences of "A" and "B" in the 'beam' coordinate of an Xarray DataArray,
    using vectorized operations (no explicit for loops). Also, include the 'nod' value for the first index of each 
    consecutive occurrence of "A" in a separate 1D array.

    Parameters:
        da (xarray.DataArray): The DataArray with dimensions 'time' and 'chan', and a 'beam' coordinate.

    Returns:
        tuple: 
            - A 2D NumPy array where column 0 contains the means for "A", column 1 contains the means for "B".
            - A 1D NumPy array with the 'nod' values for the first indices of the consecutive occurrences of "A".
    """
    beam = da.beam.values
    data = da.values
    nod = da.nod.values

    # Convert 'beam' to numerical values: "A" -> 0, "B" -> 1
    beam_numeric = np.where(beam == "A", 0, 1)

    # Identify where the 'beam' changes from A to B or B to A (create a mask for changes)
    change_points = np.diff(beam_numeric) != 0
    group_ids = np.hstack([0, np.cumsum(change_points) + 1])  # Group IDs (starting from 0)

    # Calculate the start and end indices of each group
    group_starts = np.hstack([0, np.where(change_points)[0] + 1])  # Starting indices for each group
    group_ends = np.hstack([np.where(change_points)[0], len(beam) - 1])  # Ending indices for each group

    # Assign the group label to each segment
    group_labels = beam[group_starts]

    # Create an array of means for each group (along the 'time' axis)
    group_means = np.array([data[start:end + 1].mean(axis=0) for start, end in zip(group_starts, group_ends)])

    # Separate group means for "A" and "B" based on the group labels
    means_A = group_means[group_labels == "A"]
    means_B = group_means[group_labels == "B"]

    # Extract the 'nod' values for the first occurrence of each "A" group
    nod_A = nod[group_starts[group_labels == "A"]]

    # Pad the shorter array with NaNs to ensure equal lengths
    max_length = max(len(means_A), len(means_B))
    means_A = np.pad(means_A, ((0, max_length - len(means_A)), (0, 0)), constant_values=np.nan)
    means_B = np.pad(means_B, ((0, max_length - len(means_B)), (0, 0)), constant_values=np.nan)

    # Stack the means for "A" and "B" side by side
    means = np.stack((means_A, means_B), axis=-1)

    # Create a DataArray using means
    da_chops = xr.DataArray(means, dims=["chop", "chan", "beam"], 
                      coords={"chop": np.arange(means.shape[0]), 
                              "chan": np.arange(means.shape[1]), 
                              "beam": ["A", "B"]})

    # Add nods as a coordinate on 'chop'
    da_chops.coords["nod"] = ("chop", nod_A)
    da_chops = da_chops.assign_coords(chan=da['chan'])

    return da_chops



def calculate_beam_differences(da):
    """
    Calculates the mean difference between beams "A" and "B" based on "nod".

    Parameters:
        da (xarray.DataArray): The input DataArray with indexes "chop", "beam", "chan" and a "nod" coordinate.

    Returns:
        np.ndarray: A 1D array with the length of "chan", representing the mean differences.
    """
    # Mask data by nod values for "A1" or "A2" and calculate "A - B"
    nod_a_mask = (da.nod == "A1") | (da.nod == "A2")
    nod_a_data = da.where(nod_a_mask, drop=True)
    nod_a_diffs = nod_a_data.sel(beam="A") - nod_a_data.sel(beam="B")

    # Mask data by nod values for "B1" or "B2" and calculate "B - A"
    nod_b_mask = (da.nod == "B1") | (da.nod == "B2")
    nod_b_data = da.where(nod_b_mask, drop=True)
    nod_b_diffs = nod_b_data.sel(beam="B") - nod_b_data.sel(beam="A")

    # Combine the differences
    all_diffs = xr.concat([nod_a_diffs, nod_b_diffs], dim="chop")

    # Calculate the mean along the "chop" dimension
    mean_diffs = all_diffs.mean(dim="chop")

    return mean_diffs.values


def apply_function_to_list(data_arrays, func):
    """
    Applies a function to each element in a list of xarray.DataArray objects 
    and stacks the results into a 2D numpy.ndarray.

    Parameters:
        data_arrays (list of xarray.DataArray): List of DataArray objects.
        func (callable): Function that takes a DataArray as input and outputs a 1D numpy.ndarray.

    Returns:
        numpy.ndarray: A 2D array where each row is the result of applying the function to a DataArray.
    """
    results = [func(da) for da in data_arrays]
    return np.vstack(results)


import xarray as xr
import numpy as np

def combine_numpy_with_xarray(numpy_array, dataarray, dim_name="ABBA_index"):
    """
    Combines a NumPy array with an Xarray DataArray by aligning dimensions and coordinates,
    copying all coordinates along the 'chan' dimension from the DataArray, and ignoring other dimensions.

    Parameters:
        numpy_array (np.ndarray): The NumPy array to combine (must align with one of the DataArray's dimensions).
                                  Shape should be (dim_name_size, chan_size).
        dataarray (xr.DataArray): The Xarray DataArray providing coordinate information for the 'chan' dimension.
        dim_name (str): The name of the new dimension to add (default: "ABBA_index").

    Returns:
        xr.DataArray: A new DataArray with dimensions from the NumPy array and coordinates from the DataArray.
    """
    # Validate the second dimension of the NumPy array matches the 'chan' size in the DataArray
    if numpy_array.shape[1] != dataarray.sizes["chan"]:
        raise ValueError(
            "The second dimension of the NumPy array must match the size of the 'chan' dimension in the DataArray."
        )
    
    # Create a coordinate for the new dimension (using indices by default)
    dim_coord = np.arange(numpy_array.shape[0])

    # Prepare a dictionary for the new coordinates
    coords = {dim_name: dim_coord}

    # Add all coordinates associated with the 'chan' dimension
    for coord_name, coord_value in dataarray.coords.items():
        if "chan" in dataarray.coords[coord_name].dims:
            coords[coord_name] = coord_value

    # Combine into a new DataArray with the appropriate dimensions and coordinates
    combined = xr.DataArray(
        numpy_array,
        dims=(dim_name, "chan"),
        coords=coords,
    )
    return combined


def ABBA_chop_list_to_spectra(da_abba_chop_list):
    np_spectrum_per_ABBA = apply_function_to_list(da_abba_chop_list, calculate_beam_differences)
    da_spectrum_per_ABBA = combine_numpy_with_xarray(np_spectrum_per_ABBA, da_abba_chop_list[0], dim_name="ABBA_index")

    return da_spectrum_per_ABBA


def mean_along_ABBA_index(da_ABBA_spectra):
    """
    Calculate the mean along the 'ABBA_index' dimension in an Xarray DataArray
    while preserving all coordinates.
    
    Parameters:
        da_ABBA_spectra (xr.DataArray): Input DataArray with dimensions 'ABBA_index' and 'chan'.
    
    Returns:
        xr.DataArray: DataArray with the mean calculated along the 'ABBA_index' dimension.
    """
    # Calculate the mean along "ABBA_index"
    mean_da = da_ABBA_spectra.mean(dim="ABBA_index", keep_attrs=True)
    
    return mean_da