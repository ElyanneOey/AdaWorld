"""
Extract frame pairs around a chosen action (default: 'jump').

Walks video_dir looking for episodes that each contain:
  frames.mp4   -- the video
  actions.json -- list of per-frame action dicts, e.g. [{"desc": "jump"}, ...]

For every occurrence of the action, saves a side-by-side PNG:
  LEFT  = frame when the action happened
  RIGHT = frame N frames later (default: 10)

Output structure:
  out_dir/
    per_game/
      <game>/
        <seed>_<episode>_f000042.png
        ...
    random_100/          <- only with --random-sample
      <game>_<seed>_<ep>_f000042.png
      ...

Usage
-----
python New_stuff/extract_jump_frames.py \
    --video-dir /scratch-shared/scur0531/skipped_frames_v0.0.0 \
    --out-dir ./jump_frames/skipped \
    --random-sample 100 \
    --no-per-game
"""

import argparse
import glob
import json
import os
import random as _random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


# ========================== Helpers =========================================

def _to_action_key(action):
    if action is None:
        return None
    if isinstance(action, dict):
        return action.get('desc', action.get('description', str(sorted(action.items()))))
    try:
        return str(tuple(int(x) for x in action))
    except (TypeError, ValueError):
        return str(action)


def _load_frames_from_video(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _video_frame_count(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _clean_name(name):
    import re
    name = re.sub(r'^retro_', '', name)
    name = re.sub(r'_v[\d.]+$', '', name)
    return name


# ========================== Save pair =======================================

def _make_info_strip(width, text, strip_height=22):
    strip = Image.new('RGB', (width, strip_height), color=(240, 240, 240))
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 14)
    except Exception:
        font = ImageFont.load_default()
    draw.text((6, 3), text, fill=(0, 0, 0), font=font)
    return np.array(strip)


def save_pair(frame_at, frame_later, save_path, info_text, frame_offset):
    h1, w1 = frame_at.shape[:2]
    h2, w2 = frame_later.shape[:2]
    if h1 != h2:
        frame_later = np.array(Image.fromarray(frame_later).resize((w2, h1), Image.BILINEAR))
    sep = np.full((h1, 6, 3), 160, dtype=np.uint8)
    pair = np.concatenate([frame_at, sep, frame_later], axis=1)
    info = _make_info_strip(pair.shape[1], f'at {info_text}   |   +{frame_offset} frames')
    composite = np.concatenate([info, pair], axis=0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(composite).save(save_path)


# ========================== Candidate collection ============================

def collect_candidates(video_dir, action_name, frame_offset, list_actions=False):
    """Walk video_dir, read actions.json per episode, collect jump-event metadata.

    Expected layout: video_dir/<game>/<seed>/<episode>/frames.mp4
                                                       actions.json

    actions.json format:
      {
        "action_descriptions": ["noop", "right", "left", "none", "crouch", "jump", ...],
        "actions": [{"src_id": 0, "tgt_id": 1, "action": [0,0,0,0,1,0], ...}, ...]
      }

    action_descriptions[0] is "noop" (implicit, not in the action vector).
    action_descriptions[1:] map to action vector indices 0, 1, 2, ...
    """
    video_files = sorted(glob.glob(
        os.path.join(video_dir, '**', 'frames.mp4'), recursive=True))
    if not video_files:
        raise RuntimeError(f'No frames.mp4 files found under {video_dir}')

    print(f'Found {len(video_files)} episodes to scan...')
    candidates = []
    all_action_keys = set()
    first = True

    for video_path in video_files:
        episode_dir = os.path.dirname(video_path)
        actions_json_path = os.path.join(episode_dir, 'actions.json')
        if not os.path.isfile(actions_json_path):
            continue

        with open(actions_json_path) as f:
            data = json.load(f)

        # action_descriptions[0] = "noop" (not in vector), [1:] map to vector indices
        descriptions = data.get('action_descriptions', [])
        desc_to_vec_idx = {desc: i for i, desc in enumerate(descriptions[1:])}
        frame_entries = data.get('actions', [])

        if first and descriptions:
            print(f'  [debug] action_descriptions: {descriptions}')
            first = False

        parts = Path(video_path).relative_to(video_dir).parts
        if len(parts) < 3:
            continue
        game, seed, episode = parts[0], parts[1], parts[2]
        n_frames = _video_frame_count(video_path)

        for entry in frame_entries:
            vec = entry.get('action', [])
            src_id = entry.get('src_id', -1)

            pressed = [desc for desc, idx in desc_to_vec_idx.items()
                       if idx < len(vec) and vec[idx] == 1]
            label = '+'.join(sorted(pressed)) if pressed else 'noop'
            all_action_keys.add(label)

            if list_actions:
                continue

            if action_name not in label.split('+'):
                continue
            if src_id < 0 or src_id + frame_offset >= n_frames:
                continue

            candidates.append({
                'game':       game,
                'game_clean': _clean_name(game),
                'seed':       seed,
                'episode':    episode,
                'video_path': video_path,
                'frame_idx':  src_id,
            })

    if list_actions:
        print(f'\nAll unique action labels found ({len(all_action_keys)}):')
        for k in sorted(all_action_keys):
            print(f'  {k}')
        return []

    print(f'Unique action labels seen: {sorted(all_action_keys)[:20]}')
    print(f'Found {len(candidates)} "{action_name}" events across '
          f'{len({c["game"] for c in candidates})} games')
    return candidates


# ========================== Saving ==========================================

def save_per_game(candidates, out_dir, frame_offset, max_per_game):
    by_video = defaultdict(list)
    for c in candidates:
        by_video[c['video_path']].append(c)

    counts_per_game = defaultdict(int)
    total = 0

    for video_path, items in by_video.items():
        to_save = [c for c in items
                   if not max_per_game or counts_per_game[c['game']] < max_per_game]
        if not to_save:
            continue

        frames = _load_frames_from_video(video_path)

        for c in to_save:
            if max_per_game and counts_per_game[c['game']] >= max_per_game:
                continue
            later = c['frame_idx'] + frame_offset
            if later >= len(frames):
                continue
            fname = f"{c['seed']}_{c['episode']}_f{c['frame_idx']:06d}.png"
            save_path = os.path.join(out_dir, 'per_game', c['game_clean'], fname)
            info_text = f"{c['game_clean']}  frame {c['frame_idx']}"
            save_pair(frames[c['frame_idx']], frames[later], save_path, info_text, frame_offset)
            counts_per_game[c['game']] += 1
            total += 1

    print(f'Per-game: saved {total} pairs → {os.path.join(out_dir, "per_game")}/')


def save_random_sample(candidates, out_dir, frame_offset, n_samples, rng_seed):
    """Sample n_samples spread evenly across games, save to a flat folder."""
    by_game = defaultdict(list)
    for c in candidates:
        by_game[c['game']].append(c)

    rng = _random.Random(rng_seed)
    per_game = max(1, n_samples // len(by_game))

    selected = []
    for items in by_game.values():
        selected.extend(rng.sample(items, min(per_game, len(items))))

    # Top up to n_samples if short
    if len(selected) < n_samples:
        already = set(id(c) for c in selected)
        remaining = [c for c in candidates if id(c) not in already]
        rng.shuffle(remaining)
        selected.extend(remaining[:n_samples - len(selected)])

    rng.shuffle(selected)
    selected = selected[:n_samples]

    # Load each video only once
    by_video = defaultdict(list)
    for c in selected:
        by_video[c['video_path']].append(c)

    sample_dir = os.path.join(out_dir, f'random_{n_samples}')
    total = 0

    for video_path, items in by_video.items():
        frames = _load_frames_from_video(video_path)
        for c in items:
            later = c['frame_idx'] + frame_offset
            if later >= len(frames):
                continue
            fname = f"{c['game_clean']}_{c['seed']}_{c['episode']}_f{c['frame_idx']:06d}.png"
            save_path = os.path.join(sample_dir, fname)
            info_text = f"{c['game_clean']}  frame {c['frame_idx']}"
            save_pair(frames[c['frame_idx']], frames[later], save_path, info_text, frame_offset)
            total += 1

    print(f'Random sample: saved {total} pairs → {sample_dir}/')


# ========================== Main ============================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--video-dir', type=str, required=True,
                   help='Root of the video data (contains <game>/<seed>/<episode>/frames.mp4)')
    p.add_argument('--out-dir', type=str, default='./jump_frames',
                   help='Where to save the frame pairs (default: ./jump_frames)')
    p.add_argument('--action', type=str, default='jump',
                   help='Action label to search for (default: jump)')
    p.add_argument('--frame-offset', type=int, default=10,
                   help='Frames after the action to show on the right (default: 10)')
    p.add_argument('--max-per-game', type=int, default=None,
                   help='Max pairs per game in per_game folder (default: unlimited)')
    p.add_argument('--random-sample', type=int, default=None,
                   help='Also save this many randomly sampled pairs in a flat folder')
    p.add_argument('--random-seed', type=int, default=42,
                   help='Random seed for --random-sample (default: 42)')
    p.add_argument('--no-per-game', action='store_true',
                   help='Skip per-game saving, only save the random sample')
    p.add_argument('--list-actions', action='store_true',
                   help='Print all unique action labels found and exit (useful for debugging)')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f'Looking for action "{args.action}", offset +{args.frame_offset} frames...\n')

    candidates = collect_candidates(args.video_dir, args.action, args.frame_offset,
                                    list_actions=args.list_actions)
    if args.list_actions:
        return

    if not candidates:
        print('No candidates found — check your path and action label.')
        return

    if not args.no_per_game:
        save_per_game(candidates, args.out_dir, args.frame_offset, args.max_per_game)

    if args.random_sample:
        save_random_sample(candidates, args.out_dir, args.frame_offset,
                           args.random_sample, args.random_seed)

    print(f'\nDone. Output → {args.out_dir}/')


if __name__ == '__main__':
    main()
