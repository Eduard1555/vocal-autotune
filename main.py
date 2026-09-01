import numpy as np
import librosa
import soundfile as sf
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import librosa.display
import os

root = tk.Tk()
root.withdraw()

audio_path = filedialog.askopenfilename(
    title="Alege un fisier audio",
    filetypes=[
        ("Audio Files", "*.wav *.mp3 *.flac *.ogg *.m4a"),
        ("WAV Files", "*.wav"),
        ("MP3 Files", "*.mp3"),
        ("All Files", "*.*")
    ]
)

if not audio_path:
    raise ValueError("Nu ai selectat niciun fisier audio.")

print(f"Fisier selectat: {audio_path}")

signal, sr = librosa.load(audio_path, sr=None, mono=True)

print("=== AUDIO INPUT ===")
print(f"Sample rate: {sr} Hz")
print(f"Numar esantioane: {len(signal)}")
print(f"Durata: {len(signal) / sr:.2f} secunde")

signal = signal - np.mean(signal)

max_val = np.max(np.abs(signal))
if max_val > 0:
    signal = signal / max_val

signal_trimmed, index = librosa.effects.trim(signal, top_db=20)

print("\n=== PREPROCESSING ===")
print(f"Semnal initial: {len(signal)} esantioane")
print(f"Semnal dupa trim: {len(signal_trimmed)} esantioane")

sf.write("preprocessed_audio.wav", signal_trimmed, sr)

fmin = librosa.note_to_hz("C2")
fmax = librosa.note_to_hz("C6")

frame_length = 2048
hop_length = 512

print("\n=== FRAME SEGMENTATION ===")
print(f"Frame length: {frame_length}")
print(f"Hop length: {hop_length}")

f0, voiced_flag, voiced_prob = librosa.pyin(
    signal_trimmed,
    fmin=fmin,
    fmax=fmax,
    sr=sr,
    frame_length=frame_length,
    hop_length=hop_length
)

print("\n=== PITCH DETECTION ===")
print(f"Numar valori F0 detectate: {len(f0)}")
print(f"Frame-uri vocale detectate: {np.sum(voiced_flag)}")

mapped_notes = []
mapped_frequencies = []

for i, freq in enumerate(f0):
    if voiced_flag[i] and np.isfinite(freq) and freq > 0:
        nearest_note = librosa.hz_to_note(freq)
        nearest_note_freq = librosa.note_to_hz(nearest_note)
    else:
        nearest_note = "No note"
        nearest_note_freq = 0.0

    mapped_notes.append(nearest_note)
    mapped_frequencies.append(nearest_note_freq)

mapped_frequencies = np.array(mapped_frequencies)

print("\n=== MUSICAL NOTE MAPPING ===")
print("Primele 10 mapari:")
for i in range(min(10, len(f0))):
    f0_str = f"{f0[i]:.2f}" if np.isfinite(f0[i]) else "nan"
    print(
        f"Frame {i}: F0 = {f0_str} Hz | "
        f"Nota = {mapped_notes[i]} | "
        f"Target = {mapped_frequencies[i]:.2f} Hz"
    )

correction_strength = 1
min_correction_steps = 0.0
max_correction_steps = 4.0
min_voiced_probability = 0.4

all_n_steps = []

for i in range(len(f0)):
    f0_val = f0[i]
    target_freq = mapped_frequencies[i]

    valid_pitch = (
        voiced_flag[i] and
        voiced_prob[i] >= min_voiced_probability and
        np.isfinite(f0_val) and
        f0_val > 0 and
        target_freq > 0
    )

    if not valid_pitch:
        all_n_steps.append(0.0)
    else:
        full_n_steps = 12 * np.log2(target_freq / f0_val)

        if abs(full_n_steps) < min_correction_steps:
            all_n_steps.append(0.0)
        else:
            n_steps = correction_strength * full_n_steps
            n_steps = np.clip(n_steps, -max_correction_steps, max_correction_steps)
            all_n_steps.append(float(n_steps))

all_n_steps = np.array(all_n_steps, dtype=float)

smoothing_window = 4

if len(all_n_steps) >= smoothing_window:
    kernel = np.ones(smoothing_window) / smoothing_window
    smoothed_n_steps = np.convolve(all_n_steps, kernel, mode="same")
else:
    smoothed_n_steps = all_n_steps.copy()

print("\n=== CORECTIE PITCH ===")
print(f"Correction strength: {correction_strength * 100:.0f}%")
print(f"Prag minim corectie: {min_correction_steps:.2f} semitonuri")
print(f"Corectie maxima: +/- {max_correction_steps:.2f} semitonuri")
print(f"Probabilitate minima voiced: {min_voiced_probability}")
print(f"Fereastra smoothing: {smoothing_window} frame-uri")
print("Primele 10 valori n_steps:")
print(smoothed_n_steps[:10])

valid_steps = np.abs(smoothed_n_steps[np.abs(smoothed_n_steps) > 1e-6])

print("\n=== STATISTICI CORECTIE ===")
if len(valid_steps) > 0:
    print(f"Corectie medie aplicata: {np.mean(valid_steps):.4f} semitonuri")
    print(f"Corectie maxima aplicata: {np.max(valid_steps):.4f} semitonuri")
else:
    print("Nu s-a aplicat corectie semnificativa.")

print("\n=== PITCH SHIFTING ===")

output_signal = np.zeros(len(signal_trimmed))
overlap_count = np.zeros(len(signal_trimmed))

for i in range(len(f0)):
    start = i * hop_length
    end = start + frame_length

    if start >= len(signal_trimmed):
        break

    frame = signal_trimmed[start:min(end, len(signal_trimmed))]

    if len(frame) < 32:
        continue

    n_steps = smoothed_n_steps[i]

    if abs(n_steps) < 1e-6:
        shifted_frame = frame.copy()
    else:
        n_fft = 2 ** int(np.floor(np.log2(len(frame))))
        n_fft = max(n_fft, 32)

        shifted_frame = librosa.effects.pitch_shift(
            frame,
            sr=sr,
            n_steps=float(n_steps),
            n_fft=n_fft
        )

    window = np.hanning(len(shifted_frame))
    shifted_frame = shifted_frame * window

    out_end = start + len(shifted_frame)

    if out_end > len(output_signal):
        shifted_frame = shifted_frame[:len(output_signal) - start]
        window = window[:len(shifted_frame)]
        out_end = len(output_signal)

    output_signal[start:out_end] += shifted_frame
    overlap_count[start:out_end] += window

overlap_count_safe = np.where(overlap_count > 0, overlap_count, 1.0)
output_signal = output_signal / overlap_count_safe

out_max = np.max(np.abs(output_signal))
if out_max > 0:
    output_signal = output_signal / out_max

print("Pitch shifting finalizat.")

output_path = "autotune_output.wav"
sf.write(output_path, output_signal.astype(np.float32), sr)

print("\n=== OUTPUT AUDIO ===")
print(f"Fisier salvat: {output_path}")

print("\n=== EVALUARE: PITCH ERROR ===")

f0_after, voiced_flag_after, voiced_prob_after = librosa.pyin(
    output_signal,
    fmin=fmin,
    fmax=fmax,
    sr=sr,
    frame_length=frame_length,
    hop_length=hop_length
)

min_len = min(len(f0), len(f0_after), len(mapped_frequencies))

f0_before_eval = f0[:min_len]
f0_after_eval = f0_after[:min_len]
target_eval = mapped_frequencies[:min_len]

valid_before = (
    np.isfinite(f0_before_eval) &
    (f0_before_eval > 0) &
    np.isfinite(target_eval) &
    (target_eval > 0) &
    voiced_flag[:min_len]
)

valid_after = (
    np.isfinite(f0_after_eval) &
    (f0_after_eval > 0) &
    np.isfinite(target_eval) &
    (target_eval > 0) &
    voiced_flag_after[:min_len]
)

pitch_error_before = np.abs(
    12 * np.log2(f0_before_eval[valid_before] / target_eval[valid_before])
)

pitch_error_after = np.abs(
    12 * np.log2(f0_after_eval[valid_after] / target_eval[valid_after])
)

pitch_error_before_clean = pitch_error_before[pitch_error_before <= 2.0]
pitch_error_after_clean = pitch_error_after[pitch_error_after <= 2.0]

if len(pitch_error_before_clean) > 0 and len(pitch_error_after_clean) > 0:
    mean_before = np.mean(pitch_error_before_clean)
    mean_after = np.mean(pitch_error_after_clean)

    improvement = mean_before - mean_after
    improvement_percent = (improvement / mean_before) * 100 if mean_before > 0 else 0

    print("\n--- Eroare pitch in semitonuri ---")
    print(f"Eroare medie inainte: {mean_before:.4f}")
    print(f"Eroare medie dupa: {mean_after:.4f}")

    print("\n--- Rezultat evaluare ---")
    print(f"Imbunatatire medie: {improvement:.4f} semitonuri")
    print(f"Imbunatatire procentuala: {improvement_percent:.2f}%")

    threshold = 0.1

    correct_before = np.sum(pitch_error_before_clean < threshold) / len(pitch_error_before_clean)
    correct_after = np.sum(pitch_error_after_clean < threshold) / len(pitch_error_after_clean)

    print("\n--- PROCENT FRAME-URI CORECTE ---")
    print(f"Inainte: {correct_before * 100:.2f}%")
    print(f"Dupa: {correct_after * 100:.2f}%")

    if mean_after < mean_before:
        conclusion = "Concluzie: autotune-ul a redus eroarea medie de pitch."
    else:
        conclusion = "Concluzie: autotune-ul nu a redus eroarea medie de pitch."

    print(conclusion)

else:
    print("Nu exista suficiente frame-uri valide pentru evaluare.")

print("\n=== LOG SPECTRAL DISTANCE ===")

min_audio_len = min(len(signal_trimmed), len(output_signal))

original_eval = signal_trimmed[:min_audio_len]
output_eval = output_signal[:min_audio_len]

S_original = np.abs(
    librosa.stft(
        original_eval,
        n_fft=2048,
        hop_length=512
    )
)

S_output = np.abs(
    librosa.stft(
        output_eval,
        n_fft=2048,
        hop_length=512
    )
)

epsilon = 1e-10

log_original = 20 * np.log10(S_original + epsilon)
log_output = 20 * np.log10(S_output + epsilon)

lsd_per_frame = np.sqrt(
    np.mean((log_original - log_output) ** 2, axis=0)
)

lsd_mean = np.mean(lsd_per_frame)

print(f"Log Spectral Distance mediu: {lsd_mean:.4f} dB")

frame_times = librosa.frames_to_time(np.arange(min_len), sr=sr, hop_length=hop_length)

err_b = np.full(min_len, np.nan)
err_b[valid_before] = pitch_error_before

err_a = np.full(min_len, np.nan)
err_a[valid_after] = pitch_error_after

W = 20
def rolling_mean(arr, w):
    out = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        chunk = arr[max(0, i - w // 2): i + w // 2]
        valid = chunk[~np.isnan(chunk)]
        if len(valid) > 0:
            out[i] = np.mean(valid)
    return out

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(frame_times, rolling_mean(err_b, W), color="red", linewidth=1.5,
        label=f"Before ({mean_before:.3f} st.)")
ax.plot(frame_times, rolling_mean(err_a, W), color="blue", linewidth=1.5,
        label=f"After ({mean_after:.3f} st.)")

ax.set_title("Pitch Error: Before vs. After Autotune")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Pitch error (semitones)")
ax.set_ylim([0, 0.8])
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()