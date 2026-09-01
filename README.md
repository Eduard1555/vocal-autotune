# Automatic Pitch Correction / Autotune

A Python project that performs automatic pitch correction on an audio file.

The program analyzes the input audio, detects the fundamental frequency (F0), maps detected pitches to the nearest musical notes, calculates the required pitch correction and applies pitch shifting to produce an autotuned output.

## Features

- Audio file selection through a graphical file dialog
- Audio preprocessing and silence trimming
- Fundamental frequency detection using `librosa.pyin`
- Musical note mapping
- Automatic pitch correction in semitones
- Pitch shifting using `librosa`
- Smoothing of pitch corrections
- Output audio generation
- Pitch-error evaluation before and after correction
- Log Spectral Distance calculation
- Pitch-error visualization

## Technologies

- Python
- NumPy
- Librosa
- SoundFile
- Tkinter
- Matplotlib

## How It Works

1. Select an audio file (`.wav`, `.mp3`, `.flac`, `.ogg` or `.m4a`).
2. The audio is loaded and preprocessed.
3. The fundamental frequency is detected for each frame.
4. Each detected pitch is mapped to the nearest musical note.
5. The required correction is calculated in semitones.
6. Pitch shifting is applied to the audio frames.
7. The corrected audio is saved as `autotune_output.wav`.
8. The result is evaluated by comparing pitch error before and after correction.

## Running

Install the required dependencies:

```bash
pip install numpy librosa soundfile matplotlib
```

Then run:

```bash
python main.py
```

Select an audio file when the file dialog appears.

## Output

The program generates:

```text
preprocessed_audio.wav
autotune_output.wav
```

It also displays pitch-correction statistics and a graph comparing pitch error before and after autotune.

## Correction Settings

The current implementation uses:

- Correction strength: 100%
- Maximum correction: ±4 semitones
- Minimum voiced probability: 0.4
- Smoothing window: 4 frames

These parameters can be adjusted directly in `main.py`.
