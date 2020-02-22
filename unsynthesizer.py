#!/usr/bin/env python
"""Main file for MakeMIT 2020 project: an "acoustic keyboard"."""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

import aubio
import pyaudio

SAMPLE_RATE: int = 44100
CHUNK_SIZE: int = 4096
CHUNK_SECONDS: float = CHUNK_SIZE / SAMPLE_RATE
HOP_SIZE: int = 512
DETECTION_THRESHOLD: float = 0.1
MIN_FREQ: float = 3000  # TODO: remove


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

    figure, axes = plt.subplots()
    fig_id = figure.number
    axes.set_xlim(0, CHUNK_SECONDS)
    axes.set_ylim(0, 20000)
    axes.set_xlabel("Time (s)")
    axes.set_ylabel("Frequency (Hz)")

    pitch_detector = aubio.pitch(  # nolint pyre-ignore pylint:disable=no-member
        buf_size=CHUNK_SIZE, hop_size=HOP_SIZE, samplerate=SAMPLE_RATE
    )
    onset_detector = aubio.onset(  # nolint pyre-ignore pylint:disable=no-member
        method="phase",
        buf_size=CHUNK_SIZE,
        hop_size=HOP_SIZE,
        samplerate=SAMPLE_RATE,
    )

    def step() -> None:
        data_bytes = stream.read(CHUNK_SIZE)
        n_vals = len(data_bytes) // 4
        if n_vals == 0:
            raise KeyboardInterrupt

        signal_chunk = np.frombuffer(
            data_bytes, dtype=aubio.float_type, count=n_vals
        )
        hops = signal_chunk.reshape([-1, HOP_SIZE])

        # Detect onsets and notes
        pitches = [pitch_detector(hop)[0] for hop in hops]
        onsets = [onset_detector(hop)[0] for hop in hops]

        for onset, pitch in zip(onsets, pitches):
            if onset > DETECTION_THRESHOLD and pitch > MIN_FREQ:
                print(onset, pitch)

        # Plot spectrogram
        frequencies, times, spectrogram = scipy.signal.spectrogram(
            signal_chunk, SAMPLE_RATE
        )
        axes.clear()
        axes.pcolormesh(times, frequencies, spectrogram)
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
