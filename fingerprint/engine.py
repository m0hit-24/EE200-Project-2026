import os
import glob
import pickle
from collections import defaultdict, Counter

import numpy as np
import scipy.signal as sps
import librosa
from scipy.ndimage import maximum_filter


# ──────────────────────────────────────────────────────────────────
# 1. Spectrogram
# ──────────────────────────────────────────────────────────────────
def compute_spectrogram(y, sr, nperseg=2048, noverlap=None):
        if noverlap is None:
        noverlap = nperseg // 2
    f, t, Zxx = sps.stft(y, fs=sr, window='hann', nperseg=nperseg, noverlap=noverlap)
    S = np.abs(Zxx)
    S_dB = 20 * np.log10(S + 1e-6)
    return f, t, S_dB


# ──────────────────────────────────────────────────────────────────
# 2. Peak picking (constellation map)
# ──────────────────────────────────────────────────────────────────
def find_peaks_2d(S_dB, amp_threshold_db=-35, neighborhood=(20, 20)):
    local_max = maximum_filter(S_dB, size=neighborhood) == S_dB
    above_thresh = S_dB > amp_threshold_db
    peak_mask = local_max & above_thresh
    freq_idx, time_idx = np.nonzero(peak_mask)
    return list(zip(freq_idx, time_idx))


# ──────────────────────────────────────────────────────────────────
# 3. Hash generation (peak pairing)
# ──────────────────────────────────────────────────────────────────
def generate_hashes(peaks, f, t, fan_out=5, min_dt=0.0, max_dt=2.0):
    peaks_sorted = sorted(peaks, key=lambda p: p[1])
    hashes = []

    for i, (f1_idx, t1_idx) in enumerate(peaks_sorted):
        t1, f1 = t[t1_idx], f[f1_idx]
        count = 0
        for j in range(i + 1, len(peaks_sorted)):
            f2_idx, t2_idx = peaks_sorted[j]
            t2 = t[t2_idx]
            dt = t2 - t1
            if dt < min_dt:
                continue
            if dt > max_dt:
                break  # peaks are time-sorted, so later ones are even further away
            f2 = f[f2_idx]
            key = (int(round(f1)), int(round(f2)), int(round(dt * 1000)))  # dt in ms
            hashes.append((key, t1))
            count += 1
            if count >= fan_out:
                break
    return hashes


# ──────────────────────────────────────────────────────────────────
# 4. Fingerprint a single file (used for both songs and queries)
# ──────────────────────────────────────────────────────────────────
def fingerprint_file(path, sr=22050, nperseg=2048, amp_threshold_db=-35,
                      neighborhood=(20, 20), fan_out=5, max_dt=2.0):
    y, sr_ = librosa.load(path, sr=sr, mono=True)
    f, t, S_dB = compute_spectrogram(y, sr_, nperseg=nperseg)
    peaks = find_peaks_2d(S_dB, amp_threshold_db=amp_threshold_db, neighborhood=neighborhood)
    hashes = generate_hashes(peaks, f, t, fan_out=fan_out, max_dt=max_dt)
    return hashes, peaks, f, t, S_dB


# ──────────────────────────────────────────────────────────────────
# 5. Database building, saving, loading
# ──────────────────────────────────────────────────────────────────
def build_database(song_dir, **kwargs):

    database = defaultdict(list)
    song_files = sorted(glob.glob(os.path.join(song_dir, '*')))

    for path in song_files:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            hashes, *_ = fingerprint_file(path, **kwargs)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue
        for key, anchor_time in hashes:
            database[key].append((name, anchor_time))

    return database


def save_database(database, path='database.pkl'):
    with open(path, 'wb') as fh:
        pickle.dump(dict(database), fh)


def load_database(path='database.pkl'):
    with open(path, 'rb') as fh:
        return pickle.load(fh)


# ──────────────────────────────────────────────────────────────────
# 6. Identification (offset-histogram matching)
# ──────────────────────────────────────────────────────────────────
def identify(path, database, min_votes=5, **kwargs):
    hashes, peaks, f, t, S_dB = fingerprint_file(path, **kwargs)

    offsets = defaultdict(Counter)
    for key, q_time in hashes:
        if key in database:
            for song_name, song_time in database[key]:
                offset = round(song_time - q_time, 1)
                offsets[song_name][offset] += 1

    scores = {}
    for song_name, counter in offsets.items():
        best_offset, best_count = counter.most_common(1)[0]
        scores[song_name] = best_count

    best_song = max(scores, key=scores.get) if scores else None
    if best_song is not None and scores[best_song] < min_votes:
        best_song = None

    return {
        'prediction': best_song,
        'scores': scores,
        'offsets': offsets,
        'peaks': peaks,
        'f': f, 't': t, 'S_dB': S_dB,
    }
