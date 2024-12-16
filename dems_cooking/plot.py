import decode as dc
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_beam_state(
        da: xr.DataArray,
        channel_index: int = 0
        ): 
    """Plot the timeseries of DEMS, with beam, state, and nod coordinates indicated (if available).
    Args:
        da: DEMS DataArray to be plotted.
        channel_index: The channel index to be plotted.
    """

    # Extract the relevant data for the specified channel
    values = da.sel(chan=da.chan.values[channel_index]).values.flatten()
    beam = da.beam.values
    state = da.state.values
    time = da.time.values

    # Convert time to seconds since the start
    time_start = time[0]
    time_seconds = (time - time_start).astype('timedelta64[ns]').astype(int) * 1e-9

    # Define the marker styles for the beam
    marker_styles = np.where(beam == "A", '.', 'x')

    # Define the color mapping for the state
    state_colors = {
        "GRAD": "blue",
        "SCAN": "green",
        "ACC": "purple",
        "TRAN": "darkgoldenrod",
        "ON": "lightblue",
        "OFF": "lightpink"
    }
    colors = np.vectorize(state_colors.get)(state)

    # Create the figure
    plt.figure(figsize=(10, 3))

    # If the `nod` coordinate exists, color the background
    if "nod" in da.coords:
        nod = da.nod.values
        nod_colors = {
            "A2": "skyblue",
            "B1": "lightcoral",
            "B2": "indianred",
            "A1": "lightblue"
        }

        for nod_value, color in nod_colors.items():
            nod_indices = np.where(nod == nod_value)[0]
            if len(nod_indices) > 0:
                ranges = np.split(nod_indices, np.where(np.diff(nod_indices) != 1)[0] + 1)
                for r in ranges:
                    plt.axvspan(
                        time_seconds[r[0]],
                        time_seconds[r[-1]],
                        color=color,
                        alpha=0.3
                    )

    # Create the scatter plot
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
    plt.legend(combined_handles, combined_labels, loc="upper left", bbox_to_anchor=(1, 1))

    # Add labels and grid
    plt.xlabel("Time (s)")
    plt.ylabel("Tsky (K)")
    plt.title(f"obsid: {da.aste_obs_id} | obstable: {da.aste_obs_file} | channel {da.chan.values[channel_index]} | F = {np.round(da.frequency.values[channel_index])} GHz")
    plt.tight_layout()

    plt.grid(True)
    
    return
