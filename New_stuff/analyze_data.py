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


# ========================== Helpers =========================================

def _to_action_key(action, is_dump2=False):
    """Convert any action format to a human-readable string key."""
    if action is None:
        return None
    if isinstance(action, dict):
        # latent_actions_dump_2 (p2p): dicts with 'action' key
        if is_dump2 and 'action' in action:
            val = action['action']
            return val if isinstance(val, str) else str(val)
        # skipped dataset: dicts with 'desc'/'description' key
        return action.get('desc', action.get('description', str(sorted(action.items()))))
    try:
        return str(tuple(int(x) for x in action))
    except (TypeError, ValueError):
        return str(action)


def _decode_keyboard(keyboard_labels, keyboard_keys=None, mouse_buttons=None):
    """Decode p2p multi-hot keyboard_labels (+ mouse_buttons) into action strings like 'w+space+left_click'."""
    kl = keyboard_labels if isinstance(keyboard_labels, torch.Tensor) else torch.as_tensor(keyboard_labels)
    if kl.ndim == 1:
        kl = kl.unsqueeze(0)
    kl = (kl > 0)

    if keyboard_keys is not None:
        kk = keyboard_keys.tolist() if isinstance(keyboard_keys, torch.Tensor) else list(keyboard_keys)
        key_names = [str(k) for k in kk]
    else:
        key_names = None

    mb = None
    if mouse_buttons is not None:
        mb = mouse_buttons if isinstance(mouse_buttons, torch.Tensor) else torch.as_tensor(mouse_buttons)

    result = []
    for i in range(kl.shape[0]):
        row = kl[i].tolist()
        pressed = [key_names[j] if key_names else str(j) for j, v in enumerate(row) if v]
        if mb is not None and i < len(mb):
            mb_val = mb[i].item()
            if mb_val == 0:
                pressed.append('left_click')
            elif mb_val == 1:
                pressed.append('right_click')
        result.append('+'.join(sorted(pressed)) if pressed else 'none')
    return result


def _get_game_name(path, dump_dir, data, no_source=False):
    """Return game name: from .pt dict if available, else from path layout."""
    if data.get('game_name'):
        return str(data['game_name'])
    if 'latent_actions_dump_2' in dump_dir:
        # layout: <dump>/<game>/<session>/latent_actions.pt → parts[-3] is game
        abs_parts = Path(path).parts
        return abs_parts[-3] if len(abs_parts) >= 3 else 'unknown'
    rel_parts = Path(path).relative_to(dump_dir).parts
    return rel_parts[0] if no_source else rel_parts[1]


# ========================== Data loading ====================================

def load_action_stats(dump_dir: str, source: str | None = None, no_source: bool = False):
    """Return per-game action counts and total latent counts from .pt files."""
    files = sorted(glob.glob(os.path.join(dump_dir, '**', 'latent_actions.pt'), recursive=True))
    if source and not no_source:
        files = [f for f in files if Path(f).relative_to(dump_dir).parts[0] == source]
    if not files:
        raise RuntimeError(f"No latent_actions.pt files found under {dump_dir}")
    print(f"Found {len(files)} files")

    is_dump2 = 'latent_actions_dump_2' in dump_dir
    game_action_counts = defaultdict(lambda: defaultdict(int))
    game_total_counts  = defaultdict(int)

    for f in files:
        try:
            data = torch.load(f, map_location='cpu')
        except Exception as e:
            print(f"  Skipping {f}: {e}")
            continue

        game = _get_game_name(f, dump_dir, data, no_source=no_source)

        z_mu = data.get('z_mu')
        if z_mu is None:
            continue
        n = z_mu.shape[0] if hasattr(z_mu, 'shape') else len(z_mu)
        game_total_counts[game] += n

        # p2p: keyboard_labels instead of actions
        if 'keyboard_labels' in data and not data.get('actions'):
            action_keys = _decode_keyboard(
                data['keyboard_labels'], data.get('keyboard_keys'), data.get('mouse_buttons'))
            for key in action_keys:
                if key is not None:
                    game_action_counts[game][key] += 1
            continue

        actions_raw = data.get('actions')
        if actions_raw is None:
            continue
        for action in actions_raw:
            key = _to_action_key(action, is_dump2=is_dump2)
            if key is not None:
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


def plot_frame_delta_per_game_overall(game_overall_delta: dict, save_path: str) -> None:
    """Bar chart: overall mean frame delta per game, sorted descending."""
    games  = sorted(game_overall_delta, key=lambda g: game_overall_delta[g], reverse=True)
    vals   = [game_overall_delta[g] for g in games]
    labels = [g.replace('retro_', '').replace('_v0.0.0', '') for g in games]
    mean   = np.mean(vals)

    fig_w = max(10, len(games) * 0.32)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    ax.bar(range(len(games)), vals, color='darkorange', alpha=0.8)
    ax.axhline(mean, color='red', linestyle='--', label=f'Mean = {mean:.2f}')
    ax.set_xticks(range(len(games)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel('Mean pixel delta')
    ax.set_title('Overall mean frame delta per game (sorted)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_frame_delta_early_mid_late(game_early_delta: dict, game_mid_delta: dict,
                                    game_late_delta: dict, save_path: str) -> None:
    """Grouped bar chart: early / mid / late mean frame delta per game.

    Sorted by the drop from early to late (largest drop first) so games
    where the agent likely dies are at the left.
    """
    games = sorted(
        set(game_early_delta) & set(game_mid_delta) & set(game_late_delta),
        key=lambda g: game_early_delta[g] - game_late_delta[g],
        reverse=True,
    )
    labels = [g.replace('retro_', '').replace('_v0.0.0', '') for g in games]
    early  = [game_early_delta[g] for g in games]
    mid    = [game_mid_delta[g]   for g in games]
    late   = [game_late_delta[g]  for g in games]

    x = np.arange(len(games))
    w = 0.25
    fig_w = max(10, len(games) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    ax.bar(x - w, early, w, label='Early (1st third)', color='steelblue',  alpha=0.85)
    ax.bar(x,     mid,   w, label='Mid   (2nd third)', color='darkorange', alpha=0.85)
    ax.bar(x + w, late,  w, label='Late  (3rd third)', color='firebrick',  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel('Mean pixel delta')
    ax.set_title('Early / Mid / Late frame delta per game  (sorted by early→late drop)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_frame_delta_distribution(game_video_deltas: dict, save_path: str) -> None:
    """Box plot: distribution of per-video mean frame delta per game, sorted by median."""
    games = sorted(game_video_deltas, key=lambda g: np.median(game_video_deltas[g]), reverse=True)
    data  = [game_video_deltas[g] for g in games]
    labels = [g.replace('retro_', '').replace('_v0.0.0', '') for g in games]

    fig_w = max(10, len(games) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    ax.boxplot(data, labels=labels, patch_artist=True,
               boxprops=dict(facecolor='darkorange', alpha=0.6),
               medianprops=dict(color='red', linewidth=1.5))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel('Mean pixel delta per video')
    ax.set_title('Distribution of per-video frame delta per game (sorted by median)')
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
    """Compute frame delta statistics from original frames/ directories.

    Returns seven dicts:
        game_action_mean_delta : {game: {action: mean_delta}}
        game_video_deltas      : {game: [mean_delta_per_video, ...]}
        game_overall_delta     : {game: overall_mean_delta}
        game_early_delta       : {game: mean_delta over first third of frames}
        game_mid_delta         : {game: mean_delta over middle third of frames}
        game_late_delta        : {game: mean_delta over last third of frames}
    """
    files = sorted(glob.glob(os.path.join(dump_dir, '**', 'latent_actions.pt'), recursive=True))
    if source:
        files = [f for f in files if Path(f).relative_to(dump_dir).parts[0] == source]

    game_action_deltas = defaultdict(lambda: defaultdict(list))
    game_video_deltas  = defaultdict(list)
    game_early_deltas  = defaultdict(list)
    game_mid_deltas    = defaultdict(list)
    game_late_deltas   = defaultdict(list)

    for f in files:
        parts = Path(f).relative_to(dump_dir).parts
        # Retro data: source/game/seed/episode/latent_actions.pt
        if len(parts) < 4:
            continue
        game, seed, episode = parts[1], parts[2], parts[3]

        frames_dir = os.path.join(video_dir, game, seed, episode, 'frames')
        if not os.path.isdir(frames_dir):
            continue

        try:
            data = torch.load(f, map_location='cpu')
        except Exception:
            continue
        actions_raw = data.get('actions')
        if actions_raw is None:
            continue

        frames = _load_frames_from_dir(frames_dir)
        if len(frames) < 2:
            continue

        video_deltas = []
        n_pairs = min(len(frames) - 1, len(actions_raw))
        for i in range(n_pairs):
            delta = float(np.mean(np.abs(frames[i + 1] - frames[i])))
            video_deltas.append(delta)

            action = actions_raw[i]
            key = _to_action_key(action)
            if key is not None:
                game_action_deltas[game][key].append(delta)

        if video_deltas:
            game_video_deltas[game].append(float(np.mean(video_deltas)))
            n = len(video_deltas)
            t1, t2 = max(1, n // 3), max(2, 2 * n // 3)
            game_early_deltas[game].append(float(np.mean(video_deltas[:t1])))
            game_mid_deltas[game].append(float(np.mean(video_deltas[t1:t2])))
            game_late_deltas[game].append(float(np.mean(video_deltas[t2:])))

    game_action_mean_delta = {
        game: {action: float(np.mean(vals)) for action, vals in action_dict.items()}
        for game, action_dict in game_action_deltas.items()
    }
    game_overall_delta = {game: float(np.mean(vids)) for game, vids in game_video_deltas.items()}
    game_early_delta   = {game: float(np.mean(vals)) for game, vals in game_early_deltas.items()}
    game_mid_delta     = {game: float(np.mean(vals)) for game, vals in game_mid_deltas.items()}
    game_late_delta    = {game: float(np.mean(vals)) for game, vals in game_late_deltas.items()}
    print(f"Computed frame deltas for {len(game_action_mean_delta)} games")
    return game_action_mean_delta, game_video_deltas, game_overall_delta, game_early_delta, game_mid_delta, game_late_delta


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


def save_frame_delta_csv(game_action_mean_delta: dict, game_overall_delta: dict, save_path: str) -> None:
    all_actions = sorted({a for stats in game_action_mean_delta.values() for a in stats})
    all_games = sorted(set(game_action_mean_delta) | set(game_overall_delta))
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['game', 'overall_mean_delta'] + all_actions)
        for game in all_games:
            overall = f"{game_overall_delta.get(game, ''):.4f}" if game in game_overall_delta else ''
            per_action = [
                f"{game_action_mean_delta[game].get(a, ''):.4f}" if game in game_action_mean_delta and game_action_mean_delta[game].get(a) else ''
                for a in all_actions
            ]
            writer.writerow([game, overall] + per_action)
    print(f"Saved: {save_path}")


# ========================== Main ============================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dump-dir', type=str, default='./latent_actions_dump')
    p.add_argument('--source', type=str, default=None,
                   help='Source subfolder, e.g. "adaworld". Ignored when --no-source is set.')
    p.add_argument('--no-source', action='store_true',
                   help='Data has no source subfolder: dump-dir/<game>/... instead of dump-dir/<source>/<game>/...')
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
    game_action_counts, game_total_counts = load_action_stats(args.dump_dir, args.source, args.no_source)

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
        delta_stats, game_video_deltas, game_overall_delta, game_early_delta, game_mid_delta, game_late_delta = \
            compute_frame_deltas(args.dump_dir, args.video_dir, args.source)
        if not delta_stats:
            print("  No frame delta results (video layout may not match this dataset — skipping frame delta plots)")
        else:
            plot_frame_delta(delta_stats,
                             os.path.join(args.out_dir, 'frame_delta_per_action.png'))
            plot_frame_delta_per_game_overall(game_overall_delta,
                                              os.path.join(args.out_dir, 'frame_delta_overall_per_game.png'))
            plot_frame_delta_distribution(game_video_deltas,
                                          os.path.join(args.out_dir, 'frame_delta_distribution.png'))
            plot_frame_delta_early_mid_late(game_early_delta, game_mid_delta, game_late_delta,
                                            os.path.join(args.out_dir, 'frame_delta_early_mid_late.png'))
        save_frame_delta_csv(delta_stats, game_overall_delta,
                             os.path.join(args.results_dir, 'frame_delta.csv'))
    else:
        print("\nSkipping frame delta (no --video-dir provided)")

    print(f"\nDone. Plots → {args.out_dir}/  Results → {args.results_dir}/")


if __name__ == '__main__':
    main()
