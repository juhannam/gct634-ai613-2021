from collections import defaultdict
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile
import torch
import torch.nn as nn
from mido import Message, MidiFile, MidiTrack
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.util import hz_to_midi, midi_to_hz

from constants import HOP_SIZE, MIN_MIDI, SAMPLE_RATE
from dataset import allocate_batch


def evaluate(model, batch, device, save=False, save_path=None):
    metrics = defaultdict(list)
    batch = allocate_batch(batch, device)

    frame_logit, onset_logit = model(batch['audio'])

    criterion = nn.BCEWithLogitsLoss()
    frame_loss = criterion(frame_logit, batch['frame'])
    onset_loss = criterion(onset_logit, batch['onset'])
    metrics['metric/loss/frame_loss'].append(frame_loss.cpu().numpy())
    metrics['metric/loss/onset_loss'].append(onset_loss.cpu().numpy())

    for batch_idx in range(batch['audio'].shape[0]):
        frame_pred = torch.sigmoid(frame_logit[batch_idx])
        onset_pred = torch.sigmoid(onset_logit[batch_idx])

        pr, re, f1 = framewise_eval(frame_pred, batch['frame'][batch_idx])
        metrics['metric/frame/frame_precision'].append(pr)
        metrics['metric/frame/frame_recall'].append(re)
        metrics['metric/frame/frame_f1'].append(f1)

        pr, re, f1 = framewise_eval(onset_pred, batch['onset'][batch_idx])
        metrics['metric/frame/onset_precision'].append(pr)
        metrics['metric/frame/onset_recall'].append(re)
        metrics['metric/frame/onset_f1'].append(f1)

        p_est, i_est = extract_notes(onset_pred, frame_pred)
        p_ref, i_ref = extract_notes(
            batch['onset'][batch_idx], batch['frame'][batch_idx])

        scaling = HOP_SIZE / SAMPLE_RATE

        i_ref = (i_ref * scaling).reshape(-1, 2)
        p_ref = np.array([midi_to_hz(MIN_MIDI + pitch) for pitch in p_ref])
        i_est = (i_est * scaling).reshape(-1, 2)
        p_est = np.array([midi_to_hz(MIN_MIDI + pitch) for pitch in p_est])

        p, r, f, o = evaluate_notes(
            i_ref, p_ref, i_est, p_est, offset_ratio=None)
        metrics['metric/note/precision'].append(p)
        metrics['metric/note/recall'].append(r)
        metrics['metric/note/f1'].append(f)
        metrics['metric/note/overlap'].append(o)

        p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
        metrics['metric/note-with-offsets/precision'].append(p)
        metrics['metric/note-with-offsets/recall'].append(r)
        metrics['metric/note-with-offsets/f1'].append(f)
        metrics['metric/note-with-offsets/overlap'].append(o)

        if save:
            stem = Path(batch["path"][batch_idx]).stem
            if len(p_est) == 0:
                print(f'No onset detected. Skip: {stem}')
            midi_filename = Path(save_path) / (stem + '.midi')
            save_midi(midi_filename, p_est, i_est, [64] * len(p_est))

            wav_filename = Path(save_path) / (stem + '.wav')
            midi_file = pretty_midi.PrettyMIDI(str(midi_filename))
            synth_audio = midi_file.fluidsynth(fs=16000)
            soundfile.write(wav_filename, synth_audio, 16000)

    return metrics


def extract_notes(onsets, frames, onset_threshold=0.5, frame_threshold=0.5):
    """Finds the note timings based on the onsets and frames information.

    Args:
        onsets: torch.FloatTensor of shape (frames, bins)
        frames: torch.FloatTensor of shape (frames, bins)
        onset_threshold: float
        frame_threshold: float

    Returns:
        pitches: np.ndarray of bin_indices
        intervals: np.ndarray of rows containing (onset_index, offset_index)
    """
    onsets = (onsets > onset_threshold).type(torch.int).cpu()
    frames = (frames > frame_threshold).type(torch.int).cpu()
    onset_diff = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]],
                           dim=0) == 1

    pitches = []
    intervals = []

    for nonzero in onset_diff.nonzero():
        frame = nonzero[0].item()
        pitch = nonzero[1].item()

        onset = frame
        offset = frame

        while onsets[offset, pitch].item() or frames[offset, pitch].item():
            offset += 1
            if offset == onsets.shape[0]:
                break
            if (offset != onset) and onsets[offset, pitch].item():
                break

        if offset > onset:
            pitches.append(pitch)
            intervals.append([onset, offset])

    return np.array(pitches), np.array(intervals)


def framewise_eval(pred, label, threshold=0.5):
    '''Evaluates frame-wise (point-wise) evaluation.

    Args:
        pred: torch.Tensor of shape (frame, pitch)
        label: torch.Tensor of shape (frame, pitch)
    '''

    tp = torch.sum((pred >= threshold) * (label == 1)).cpu().numpy()
    fn = torch.sum((pred < threshold) * (label == 1)).cpu().numpy()
    fp = torch.sum((pred >= threshold) * (label != 1)).cpu().numpy()

    pr = tp / float(tp + fp) if (tp + fp) > 0 else 0
    re = tp / float(tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * pr * re / float(pr + re) if (pr + re) > 0 else 0

    return pr, re, f1


def save_midi(path, pitches, intervals, velocities):
    """Saves extracted notes as a MIDI file.

    Args:
        path: the path to save the MIDI file
        pitches: np.ndarray of bin_indices
        intervals: list of tuple (onset_index, offset_index)
        velocities: list of velocity values
    """
    file = MidiFile()
    track = MidiTrack()
    file.tracks.append(track)
    ticks_per_second = file.ticks_per_beat * 2.0

    events = []
    for i in range(len(pitches)):
        events.append(
            dict(type='on',
                 pitch=pitches[i],
                 time=intervals[i][0],
                 velocity=velocities[i]))
        events.append(
            dict(type='off',
                 pitch=pitches[i],
                 time=intervals[i][1],
                 velocity=velocities[i]))
    events.sort(key=lambda row: row['time'])

    last_tick = 0
    for event in events:
        current_tick = int(event['time'] * ticks_per_second)
        velocity = int(event['velocity'] * 127)
        if velocity > 127:
            velocity = 127
        pitch = int(round(hz_to_midi(event['pitch'])))
        track.append(
            Message('note_' + event['type'],
                    note=pitch,
                    velocity=velocity,
                    time=current_tick - last_tick))
        last_tick = current_tick

    file.save(path)
