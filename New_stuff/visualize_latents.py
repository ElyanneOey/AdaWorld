"""
Visualize latent action vectors from the LAM using PCA, t-SNE, and UMAP.

Loads all latent_actions.pt files from latent_actions_dump/ and produces
2D scatter plots colored by action label and game identity.

Usage
-----
python new_stuff/visualize_latents.py
python new_stuff/visualize_latents.py --dump-dir ./latent_actions_dump --out-dir ./plots
python new_stuff/visualize_latents.py --max-samples 5000  # subsample for speed
"""

import argparse
import glob
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder


# ========================== Data Loading ====================================

def load_all_latents(dump_dir: str, max_samples: int | None = None):
    """Load all latent_actions.pt files and return stacked tensors with labels."""
    files = sorted(glob.glob(os.path.join(dump_dir, '**', 'latent_actions.pt'), recursive=True))
    if not files:
        raise RuntimeError(f"No latent_actions.pt files found under {dump_dir}")
    print(f"Found {len(files)} latent_actions.pt files")

    all_z = []
    all_actions = []
    all_games = []

    for f in files:
        game_name = Path(f).parts[-4]  # e.g. retro_8eyes-nes_v0.0.0
        data = torch.load(f, map_location='cpu')

        z = data['z_mu']
        if z is None:
            continue
        if isinstance(z, torch.Tensor):
            z = z.float()
        else:
            z = torch.as_tensor(z, dtype=torch.float32)
        if z.ndim == 1:
            z = z.unsqueeze(0)

        n = z.shape[0]

        # Actions (may not always be present, may contain None values)
        actions = data.get('actions', None)
        if actions is not None:
            if isinstance(actions, torch.Tensor):
                actions = actions.tolist()
            elif not isinstance(actions, list):
                try:
                    actions = list(actions)
                except Exception:
                    actions = [None] * n
            # Replace any remaining None entries
            actions = [a if a is not None else None for a in actions]
        else:
            actions = [None] * n

        all_z.append(z)
        all_actions.extend(actions)
        all_games.extend([game_name] * n)

    all_z = torch.cat(all_z, dim=0).numpy()

    # Trim to shortest to handle any per-file length mismatches
    n_total = min(len(all_z), len(all_actions), len(all_games))
    all_z = all_z[:n_total]
    all_actions = all_actions[:n_total]
    all_games = all_games[:n_total]

    # Subsample if requested
    if max_samples is not None and n_total > max_samples:
        idx = np.random.default_rng(42).choice(n_total, max_samples, replace=False)
        all_z = all_z[idx]
        all_actions = [all_actions[i] for i in idx]
        all_games = [all_games[i] for i in idx]

    print(f"Loaded {len(all_z)} samples from {len(set(all_games))} game(s)")
    return all_z, all_actions, all_games


# ========================== Dimensionality Reduction ========================

def run_pca(z: np.ndarray, n_components: int = 2) -> np.ndarray:
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(z), pca


def run_tsne(z: np.ndarray, perplexity: int = 30) -> np.ndarray:
    # First reduce to 50 dims with PCA for speed (standard practice)
    if z.shape[1] > 50:
        z = PCA(n_components=50, random_state=42).fit_transform(z)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    return tsne.fit_transform(z)


def run_umap(z: np.ndarray, n_components: int = 2, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn is not installed. Run: pip install umap-learn")
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    return reducer.fit_transform(z)


def scatter_plot_3d(
    embedding: np.ndarray,
    labels: list,
    title: str,
    save_path: str,
    max_classes: int = 300,
) -> None:
    """3D scatter plot colored by label, saved as a static PNG."""
    from mpl_toolkits.mplot3d import Axes3D

    label_encoder = LabelEncoder()

    valid_mask = [l is not None for l in labels]
    if not any(valid_mask):
        print(f"  No labels available for {title}, skipping.")
        return

    valid_emb = embedding[valid_mask]
    valid_labels = [l for l, m in zip(labels, valid_mask) if m]

    if valid_labels and isinstance(valid_labels[0], (list, np.ndarray)):
        valid_labels = [str(tuple(int(x) for x in l)) for l in valid_labels]

    encoded = label_encoder.fit_transform(valid_labels)
    classes = label_encoder.classes_

    if len(classes) > max_classes:
        print(f"  Too many classes ({len(classes)}) for {title}, showing top {max_classes}")
        top_classes = classes[:max_classes]
        mask = np.isin(valid_labels, top_classes)
        valid_emb = valid_emb[mask]
        encoded = label_encoder.transform([l for l, m in zip(valid_labels, mask) if m])
        classes = top_classes

    n = len(classes)
    cmap = plt.cm.get_cmap('tab20', n) if n <= 20 else plt.cm.get_cmap('gist_rainbow', n)

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        valid_emb[:, 0], valid_emb[:, 1], valid_emb[:, 2],
        c=encoded, cmap=cmap, alpha=0.5, s=5
    )

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cmap(i / n), markersize=6, label=str(cls))
        for i, cls in enumerate(classes)
    ]
    ncol = max(1, n // 30)
    ax.legend(handles=legend_handles, title='Label', bbox_to_anchor=(1.05, 1),
              loc='upper left', fontsize=5, ncol=ncol)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    ax.set_zlabel('dim 3')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ========================== Plotting ========================================

def scatter_plot(
    embedding: np.ndarray,
    labels: list,
    title: str,
    save_path: str,
    max_classes: int = 300,
) -> None:
    """2D scatter plot colored by label."""
    label_encoder = LabelEncoder()

    # Filter out None labels
    valid_mask = [l is not None for l in labels]
    if not any(valid_mask):
        print(f"  No labels available for {title}, skipping.")
        return

    valid_emb = embedding[valid_mask]
    valid_labels = [l for l, m in zip(labels, valid_mask) if m]

    # Convert multi-dimensional action vectors to strings for LabelEncoder
    if valid_labels and isinstance(valid_labels[0], (list, np.ndarray)):
        valid_labels = [str(tuple(int(x) for x in l)) for l in valid_labels]

    encoded = label_encoder.fit_transform(valid_labels)
    classes = label_encoder.classes_

    if len(classes) > max_classes:
        print(f"  Too many classes ({len(classes)}) for {title}, showing top {max_classes}")
        top_classes = classes[:max_classes]
        mask = np.isin(valid_labels, top_classes)
        valid_emb = valid_emb[mask]
        encoded = label_encoder.transform([l for l, m in zip(valid_labels, mask) if m])
        classes = top_classes

    n = len(classes)
    cmap = plt.cm.get_cmap('tab20', n) if n <= 20 else plt.cm.get_cmap('gist_rainbow', n)

    _, ax = plt.subplots(figsize=(14, 8))
    ax.scatter(
        valid_emb[:, 0], valid_emb[:, 1],
        c=encoded, cmap=cmap,
        alpha=0.5, s=8, linewidths=0
    )

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cmap(i / n), markersize=6, label=str(cls))
        for i, cls in enumerate(classes)
    ]
    ncol = max(1, n // 30)
    ax.legend(handles=legend_handles, title='Label', bbox_to_anchor=(1.02, 1),
              loc='upper left', fontsize=5, ncol=ncol)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def pca_variance_plot(pca, save_path: str) -> None:
    """Plot cumulative explained variance for PCA."""
    explained = np.cumsum(pca.explained_variance_ratio_) * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(explained) + 1), explained, marker='o', markersize=4)
    ax.axhline(80, color='red', linestyle='--', label='80%')
    ax.axhline(95, color='orange', linestyle='--', label='95%')
    ax.set_xlabel('Number of PCA components')
    ax.set_ylabel('Cumulative explained variance (%)')
    ax.set_title('PCA — Cumulative Explained Variance')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ========================== Per-game analysis ================================

def run_per_game(
    z: np.ndarray,
    actions: list,
    games: list,
    out_dir: str,
    method: str = 'pca',
    min_samples: int = 20,
) -> None:
    """For each game, run dimensionality reduction on that game's latents only
    and save a scatter plot colored by action."""
    game_names = sorted(set(games))
    games_arr = np.array(games)
    per_game_dir = os.path.join(out_dir, 'per_game')
    os.makedirs(per_game_dir, exist_ok=True)
    print(f"\nPer-game analysis ({method}) for {len(game_names)} games → {per_game_dir}/")

    for game in game_names:
        mask = games_arr == game
        z_g = z[mask]
        actions_g = [actions[i] for i, m in enumerate(mask) if m]
        n = len(z_g)

        if n < min_samples:
            print(f"  Skipping {game} (only {n} samples)")
            continue

        # Check that at least some actions are labelled
        has_labels = any(a is not None for a in actions_g)

        save_path = os.path.join(per_game_dir, f"{game}_{method}.png")

        try:
            if method == 'pca':
                emb, _ = run_pca(z_g, n_components=2)
            elif method == 'tsne':
                perp = min(30, n // 3)
                emb = run_tsne(z_g, perplexity=perp)
            elif method == 'umap':
                n_neighbors = min(15, n - 1)
                emb = run_umap(z_g, n_components=2, n_neighbors=n_neighbors)

            title = f"{game}\n{method.upper()} — colored by action  (n={n})"
            if has_labels:
                scatter_plot(emb, actions_g, title, save_path)
            else:
                print(f"  {game}: no action labels, skipping plot")
        except Exception as e:
            print(f"  {game}: failed ({e})")


# ========================== Main ============================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dump-dir', type=str, default='./latent_actions_dump',
                   help='Root of the latent_actions_dump folder')
    p.add_argument('--out-dir', type=str, default='./plots',
                   help='Where to save the plots')
    p.add_argument('--max-samples', type=int, default=10000,
                   help='Max samples to use (subsamples for speed)')
    p.add_argument('--method', type=str, default='all',
                   choices=['pca', 'tsne', 'umap', 'all'],
                   help='Which method to run (default: all)')
    p.add_argument('--umap-3d', action='store_true',
                   help='Also run UMAP in 3D and save a 3D scatter plot')
    p.add_argument('--per-game', action='store_true',
                   help='Also run per-game analysis: for each game, visualize action '
                        'clustering within that game only')
    p.add_argument('--per-game-method', type=str, default='pca',
                   choices=['pca', 'tsne', 'umap'],
                   help='Method to use for per-game plots (default: pca)')
    p.add_argument('--min-samples', type=int, default=20,
                   help='Skip games with fewer than this many samples in per-game mode')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading latents...")
    z, actions, games = load_all_latents(args.dump_dir, max_samples=args.max_samples)
    print(f"z shape: {z.shape}")

    # --- PCA ---
    if args.method in ('pca', 'all'):
        print("\nRunning PCA...")
        pca_2d, pca_full = run_pca(z, n_components=min(z.shape[1], z.shape[0]))
        pca_variance_plot(pca_full, os.path.join(args.out_dir, 'pca_variance.png'))

        pca_2d, _ = run_pca(z, n_components=2)
        scatter_plot(pca_2d, actions, 'PCA — colored by action',
                     os.path.join(args.out_dir, 'pca_actions.png'))
        scatter_plot(pca_2d, games, 'PCA — colored by game',
                     os.path.join(args.out_dir, 'pca_games.png'))

    # --- t-SNE ---
    if args.method in ('tsne', 'all'):
        print("\nRunning t-SNE (this may take a few minutes)...")
        tsne_2d = run_tsne(z)
        scatter_plot(tsne_2d, actions, 't-SNE — colored by action',
                     os.path.join(args.out_dir, 'tsne_actions.png'))
        scatter_plot(tsne_2d, games, 't-SNE — colored by game',
                     os.path.join(args.out_dir, 'tsne_games.png'))

    # --- UMAP ---
    if args.method in ('umap', 'all'):
        print("\nRunning UMAP (2D)...")
        umap_2d = run_umap(z, n_components=2)
        scatter_plot(umap_2d, actions, 'UMAP — colored by action',
                     os.path.join(args.out_dir, 'umap_actions.png'))
        scatter_plot(umap_2d, games, 'UMAP — colored by game',
                     os.path.join(args.out_dir, 'umap_games.png'))

        if args.umap_3d:
            print("\nRunning UMAP (3D)...")
            umap_3d = run_umap(z, n_components=3)
            scatter_plot_3d(umap_3d, actions, 'UMAP 3D — colored by action',
                            os.path.join(args.out_dir, 'umap_3d_actions.png'))
            scatter_plot_3d(umap_3d, games, 'UMAP 3D — colored by game',
                            os.path.join(args.out_dir, 'umap_3d_games.png'))

    # --- Per-game ---
    if args.per_game:
        run_per_game(z, actions, games, args.out_dir,
                     method=args.per_game_method,
                     min_samples=args.min_samples)

    print(f"\nDone. Plots saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
