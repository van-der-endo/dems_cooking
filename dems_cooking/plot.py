import decode as dc
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_beam_state(
        da: xr.DataArray,
        channel_index: int = 0
        ): 
    """Plot the timeseries of DEMS, with beam and state coordinates indicated.
    Args:
        da: DEMS DataArray to be plotted.
        channel_index: The channel index to be plotted.
    """

    values = da.sel(chan=da.chan.values[channel_index]).values.flatten()  # Flatten to 1D if necessary

    # Extract the relevant data
    beam = da.beam.values
    state = da.state.values
    time = da.time.values

    # Convert time to seconds since the start
    time_start = time[0]  # Get the first time value
    time_seconds = (time - time_start).astype('timedelta64[ns]').astype(int) * 1e-9 # Convert to seconds


    # Define the marker styles for the beam
    marker_styles = np.where(beam == "A", '.', 'x')  # Use "." for points and "x" for crosses

    # Define the color mapping for the state
    state_colors = {
        "GRAD": "blue",
        "SCAN": "green",
        "ACC": "purple",
        "TRAN": "darkgoldenrod"
    }
    colors = np.vectorize(state_colors.get)(state)

    # Create the scatter plot
    plt.figure(figsize=(10, 3))
    for marker in ['.', 'x']:
        mask = marker_styles == marker
        plt.scatter(time_seconds[mask], values[mask], c=colors[mask], label=f"Beam: {marker}", marker=marker)

    # Create legend handles for state-color mapping (points only, no edges)
    state_legend_handles = [
        Line2D([0], [0], marker='.', color=color, markersize=10, label=state, linestyle='None')
        for state, color in state_colors.items()
    ]

    # Create legend handles for beam-marker mapping (points and crosses, no edges)
    beam_labels = ["A", "B"]
    beam_markers = ['.', 'x']
    beam_legend_handles = [
        Line2D([0], [0], marker=marker, color='black', markersize=10, label=label, linestyle='None')
        for marker, label in zip(beam_markers, beam_labels)
    ]

    # Combine both legends into one
    combined_handles = state_legend_handles + beam_legend_handles
    combined_labels = [handle.get_label() for handle in combined_handles]
    plt.legend(combined_handles, combined_labels,  loc="upper left", bbox_to_anchor=(1, 1))

    # Add labels and grid
    plt.xlabel("Time (s)")
    plt.ylabel("Tsky (K)")
    plt.title(f"obsid: {da.aste_obs_id} | obstable: {da.aste_obs_file} | channel {da.chan.values[channel_index]} | F = {np.round(da.frequency.values[channel_index])} GHz")
    plt.tight_layout()

    plt.grid(True)
    
    return
