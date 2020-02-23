#!/usr/bin/env python
"""Main file for MakeMIT 2020 project: an "acoustic keyboard"."""

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

import aubio
import pyaudio

# Constants
SAMPLE_RATE: int = 44100
CHUNK_SIZE: int = 6144
CHUNK_SECONDS: float = CHUNK_SIZE / SAMPLE_RATE
DETECTION_THRESHOLD: float = 0.001
FILTER_DEFS: Dict[str, Tuple[int, int]] = {
    "A": (13000, 13500),  # Or, finer: 13100 - 134000
    "B": (14500, 15500),
    "C": (17400, 17700),
    "D": (16600, 16700),
    "E": (8200, 8400),
    "F": (11500, 11700),
    "Space": (4300, 4400),  # Small steel medium
    "Enter": (2450, 2500),  # Big steel long
}
COOLDOWN: float = 0.75

# Bar chart colors
COLOR_INACTIVE = (0.39, 0.64, 1.0)
COLOR_ACTIVE = (0.83, 0.18, 0.18)


# See: https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
def make_bandpass_filter(min_freq, max_freq) -> np.ndarray:  # type: ignore
    """Make a bandpass filter tuned to a specific requency range."""
    return scipy.signal.butter(
        5,
        [min_freq * 2 / SAMPLE_RATE, max_freq * 2 / SAMPLE_RATE],
        btype="bandpass",
        output="sos",
    )


def main() -> None:
    """
    Run the main program.
    """

    audio_source = pyaudio.PyAudio()
    stream = audio_source.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )

    keys = list(FILTER_DEFS.keys())
    filters = {key: make_bandpass_filter(*FILTER_DEFS[key]) for key in keys}
    global_cooldown = 0

    # Set up some figure properties ahead of time
    figure, (ax_spectrogram, ax_barplot) = plt.subplots(
        ncols=2, figsize=(20, 10)
    )
    fig_id = figure.number
    ax_spectrogram.set_xlim(0, CHUNK_SECONDS)
    ax_spectrogram.set_ylim(0, 20000)
    ax_barplot.yaxis.tick_right()
    ax_barplot.yaxis.set_label_position("right")

    def key_powers(signal: np.ndarray):  # type: ignore
        for key in keys:
            band = scipy.signal.sosfilt(filters[key], signal)
            yield np.abs(band).mean()

    def step() -> None:
        nonlocal global_cooldown

        data_bytes = stream.read(CHUNK_SIZE)
        n_vals = len(data_bytes) // 4
        if n_vals == 0:
            raise KeyboardInterrupt

        signal_chunk = np.frombuffer(
            data_bytes, dtype=aubio.float_type, count=n_vals
        )

        powers = np.asarray(list(key_powers(signal_chunk)))

        if global_cooldown <= 0:
            max_power = powers.max()

            if max_power > DETECTION_THRESHOLD:
                global_cooldown = COOLDOWN
                key_idx = powers.argmax()
                key = keys[key_idx]
                print(f"{key}: {max_power}")
        else:
            global_cooldown -= CHUNK_SECONDS

        # Plot power chart
        ax_barplot.clear()
        colors = [
            COLOR_ACTIVE if power > DETECTION_THRESHOLD else COLOR_INACTIVE
            for power in powers
        ]
        ax_barplot.bar(
            x=list(range(len(keys))),
            height=powers,
            color=colors,
            align="center",
            tick_label=keys,
        )
        ax_barplot.hlines(
            DETECTION_THRESHOLD, -0.5, len(keys) - 0.5, colors=COLOR_ACTIVE
        )
        ax_barplot.set_xlim(-0.5, len(keys) - 0.5)
        ax_barplot.set_ylim(0, 0.005)
        ax_barplot.set_xlabel("Key", fontsize=20)
        ax_barplot.set_ylabel("Power", fontsize=20)
        ax_barplot.set_title("Filter Bank Mean Powers", fontsize=24)

        # Plot spectrogram
        frequencies, times, spectrogram = scipy.signal.spectrogram(
            signal_chunk, SAMPLE_RATE
        )
        ax_spectrogram.clear()
        ax_spectrogram.pcolormesh(
            times, frequencies, spectrogram, cmap="inferno"
        )
        ax_spectrogram.set_xlabel("Time (s)", fontsize=20)
        ax_spectrogram.set_ylabel("Frequency (Hz)", fontsize=20)
        ax_spectrogram.set_title("Moving Spectrogram", fontsize=24)

        plt.pause(0.001)

    try:
        while plt.fignum_exists(fig_id):
            step()

        plt.show()
    except KeyboardInterrupt:  # Handle exit gracefully
        figure.close()
    finally:
        stream.stop_stream()
        stream.close()
        audio_source.terminate()


if __name__ == "__main__":
    main()
