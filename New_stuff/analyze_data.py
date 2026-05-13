"""
Data analysis for latent action dumps.

Produces:
  plots/analysis/action_distribution_heatmap.png  -- action balance across all games
  plots/analysis/latent_counts.png                 -- samples per game bar chart
  plots/analysis/frame_delta_per_action.png        -- mean pixel diff per action (needs --video-dir)
  results/action_distribution.csv
  results/latent_counts.csv
  results/frame_delta.csv                          -- (if --video-dir provided)

Usage
-----
python New_stuff/analyze_data.py --dump-dir ./latent_actions_dump --source adaworld
python New_stuff/analyze_data.py --dump-dir ./latent_actions_dump --source adaworld \
    --video-dir /gpfs/home3/scur0531/random_actions_data/dataset/retro_act_v0.0.0_random
"""

import argparse
import csv
import glob
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# ========================== Data loading ====================================

def load_action_stats(dump_dir: str, source: str | None = None):
    """Return per-game action counts and total latent counts from .pt files."""
    pattern = (f'{dump_dir}/{source}/*/*/*/latent_actions.pt' if source
               else f'{dump_dir}/*/*/*/latent_actions.pt')
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"No latent_actions.pt files found with pattern: {pattern}")
    print(f"Found {len(files)} files")

    game_action_counts = defaultdict(lambda: defaultdict(int))
    game_total_counts  = defaultdict(int)

    for f in files:
        game = Path(f).parts[-4]
        data = torch.load(f, map_location='cpu')

        z_mu = data.get('z_mu')
        if z_mu is None:
            continue
        n = z_mu.shape[0] if hasattr(z_mu, 'shape') else len(z_mu)
        game_total_counts[game] += n

        actions_raw = data.get('actions')
        if actions_raw is None:
            continue
        for action in actions_raw:
            if action is None:
                continue
            try:
                key = str(tuple(int(x) for x in action))
            except (TypeError, ValueError):
                key = str(action)
            game_action_counts[game][key] += 1

    print(f"Loaded stats for {len(game_total_counts)} games")
    return game_action_counts, game_total_counts


# ========================== Plots ===========================================

def plot_action_heatmap(game_action_counts: dict, save_path: str) -> None:
    """Heatmap: rows=games, columns=actions, values=percentage of samples."""
    games = sorted(game_action_counts.keys())
    all_actions = sorted({a for counts in game_action_counts.values() for a in counts})

    data = np.zeros((len(games), len(all_actions)))
    for i, game in enumerate(games):
        total = sum(game_action_counts[game].values())
        for j, action in enumerate(all_actions):
            count = game_action_counts[game].get(action, 0)
            data[i, j] = (count / total * 100) if total > 0 else 0

    fig_h = max(6, len(games) * 0.28)
    fig_w = max(6, len(all_actions) * 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(data, aspect='auto', cmap='YlOrRd', vmin=0)
    plt.colorbar(im, ax=ax, label='% of samples', shrink=0.6)

    equal_pct = 100 / len(all_actions)
    ax.set_title(f'Action class distribution per game  (balanced = {equal_pct:.1f}% each)', fontsize=12)
    ax.set_xticks(range(len(all_actions)))
    ax.set_xticklabels(all_actions, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(games)))
    ax.set_yticklabels(
        [g.replace('retro_', '').replace('_v0.0.0', '') for g in games],
        fontsize=6
    )
    ax.set_xlabel('Action')
    ax.set_ylabel('Game')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_latent_counts(game_total_counts: dict, save_path: str) -> None:
    """Bar chart of total latent samples per game, sorted descending."""
    games  = sorted(game_total_counts, key=lambda g: game_total_counts[g], reverse=True)
    counts = [game_total_counts[g] for g in games]
    labels = [g.replace('retro_', '').replace('_v0.0.0', '') for g in games]
    mean   = np.mean(counts)

    fig_w = max(10, len(games) * 0.32)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    ax.bar(range(len(games)), counts, color='steelblue', alpha=0.8)
    ax.axhline(mean, color='red', linestyle='--', label=f'Mean = {mean:.0f}')
    ax.set_xticks(range(len(games)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel('Number of latent samples')
    ax.set_title('Latent sample count per game (sorted)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_frame_delta(delta_stats: dict, save_path: str) -> None:
    """Bar chart: mean frame delta per action, one group per game."""
    games = sorted(delta_stats.keys())
    all_actions = sorted({a for stats in delta_stats.values() for a in stats})
    n_games   = len(games)
    n_actions = len(all_actions)

    cols = min(4, n_games)
    rows = (n_games + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)

    for idx, game in enumerate(games):
        ax = axes[idx // cols][idx % cols]
        vals = [delta_stats[game].get(a, 0) for a in all_actions]
        ax.bar(range(n_actions), vals, color='darkorange', alpha=0.8)
        ax.set_xticks(range(n_actions))
        ax.set_xticklabels(all_actions, rotation=45, ha='right', fontsize=6)
        ax.set_title(game.replace('retro_', '').replace('_v0.0.0', ''), fontsize=8)
        ax.set_ylabel('Mean pixel delta', fontsize=7)

    for idx in range(n_games, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle('Mean frame-to-frame pixel change per action per game', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ========================== Frame delta computation =========================

def _load_frames_from_dir(frames_dir: str) -> list[np.ndarray]:
    """Load all images from a frames/ directory sorted alphabetically."""
    from PIL import Image as PILImage
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(frames_dir, ext)))
    paths.sort()
    frames = []
    for p in paths:
        img = np.array(PILImage.open(p).convert('RGB')).astype(np.float32)
        frames.append(img)
    return frames


def compute_frame_deltas(dump_dir: str, video_dir: str, source: str | None = None):
    """Compute mean absolute pixel difference between consecutive frames per action per game.

    Matches latent_actions.pt (for action labels) with the original frames/ directories.
    Expected structure:
        dump_dir/<source>/<game>/<seed>/<episode>/latent_actions.pt
        video_dir/<game>/<seed>/<episode>/frames/   ← directory of frame images
    """
    pattern = (f'{dump_dir}/{source}/*/*/*/latent_actions.pt' if source
               else f'{dump_dir}/*/*/*/latent_actions.pt')
    files = sorted(glob.glob(pattern))

    game_action_deltas = defaultdict(lambda: defaultdict(list))

    for f in files:
        parts = Path(f).parts
        game, seed, episode = parts[-4], parts[-3], parts[-2]

        frames_dir = os.path.join(video_dir, game, seed, episode, 'frames')
        if not os.path.isdir(frames_dir):
            continue

        data = torch.load(f, map_location='cpu')
        actions_raw = data.get('actions')
        if actions_raw is None:
            continue

        frames = _load_frames_from_dir(frames_dir)
        if len(frames) < 2:
            continue

        n_pairs = min(len(frames) - 1, len(actions_raw))
        for i in range(n_pairs):
            action = actions_raw[i]
            if action is None:
                continue
            try:
                key = str(tuple(int(x) for x in action))
            except (TypeError, ValueError):
                continue
            delta = np.mean(np.abs(frames[i + 1] - frames[i]))
            game_action_deltas[game][key].append(delta)

    game_action_mean_delta = {
        game: {action: float(np.mean(vals)) for action, vals in action_dict.items()}
        for game, action_dict in game_action_deltas.items()
    }
    print(f"Computed frame deltas for {len(game_action_mean_delta)} games")
    return game_action_mean_delta


# ========================== CSV saving ======================================

def save_action_distribution_csv(game_action_counts: dict, save_path: str) -> None:
    games = sorted(game_action_counts.keys())
    all_actions = sorted({a for counts in game_action_counts.values() for a in counts})
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['game'] + all_actions + ['total'])
        for game in games:
            total = sum(game_action_counts[game].values())
            row = [game] + [game_action_counts[game].get(a, 0) for a in all_actions] + [total]
            writer.writerow(row)
    print(f"Saved: {save_path}")


def save_latent_counts_csv(game_total_counts: dict, save_path: str) -> None:
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['game', 'n_samples'])
        for game, count in sorted(game_total_counts.items(), key=lambda x: x[1], reverse=True):
            writer.writerow([game, count])
    print(f"Saved: {save_path}")


def save_frame_delta_csv(delta_stats: dict, save_path: str) -> None:
    all_actions = sorted({a for stats in delta_stats.values() for a in stats})
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['game'] + all_actions)
        for game in sorted(delta_stats.keys()):
            row = [game] + [f"{delta_stats[game].get(a, ''):.4f}" if delta_stats[game].get(a) else ''
                            for a in all_actions]
            writer.writerow(row)
    print(f"Saved: {save_path}")


# ========================== Main ============================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dump-dir', type=str, default='./latent_actions_dump')
    p.add_argument('--source', type=str, default=None,
                   help='Source subfolder, e.g. "adaworld"')
    p.add_argument('--video-dir', type=str, default=None,
                   help='Root of original video data for frame delta computation')
    p.add_argument('--out-dir', type=str, default='./plots/analysis')
    p.add_argument('--results-dir', type=str, default='./results')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print("Loading action statistics...")
    game_action_counts, game_total_counts = load_action_stats(args.dump_dir, args.source)

    print("\nPlotting action distribution heatmap...")
    plot_action_heatmap(game_action_counts,
                        os.path.join(args.out_dir, 'action_distribution_heatmap.png'))
    save_action_distribution_csv(game_action_counts,
                                 os.path.join(args.results_dir, 'action_distribution.csv'))

    print("\nPlotting latent counts...")
    plot_latent_counts(game_total_counts,
                       os.path.join(args.out_dir, 'latent_counts.png'))
    save_latent_counts_csv(game_total_counts,
                           os.path.join(args.results_dir, 'latent_counts.csv'))

    if args.video_dir:
        print("\nComputing frame deltas (this may take a while)...")
        delta_stats = compute_frame_deltas(args.dump_dir, args.video_dir, args.source)
        plot_frame_delta(delta_stats,
                         os.path.join(args.out_dir, 'frame_delta_per_action.png'))
        save_frame_delta_csv(delta_stats,
                             os.path.join(args.results_dir, 'frame_delta.csv'))
    else:
        print("\nSkipping frame delta (no --video-dir provided)")

    print(f"\nDone. Plots → {args.out_dir}/  Results → {args.results_dir}/")


if __name__ == '__main__':
    main()
