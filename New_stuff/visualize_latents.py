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


def run_umap(z: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn is not installed. Run: pip install umap-learn")
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    return reducer.fit_transform(z)


# ========================== Plotting ========================================

def scatter_plot(
    embedding: np.ndarray,
    labels: list,
    title: str,
    save_path: str,
    max_classes: int = 20,
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

    encoded = label_encoder.fit_transform(valid_labels)
    classes = label_encoder.classes_

    if len(classes) > max_classes:
        print(f"  Too many classes ({len(classes)}) for {title}, showing top {max_classes}")
        top_classes = classes[:max_classes]
        mask = np.isin(valid_labels, top_classes)
        valid_emb = valid_emb[mask]
        encoded = label_encoder.transform([l for l, m in zip(valid_labels, mask) if m])
        classes = top_classes

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.get_cmap('tab20', len(classes))

    scatter = ax.scatter(
        valid_emb[:, 0], valid_emb[:, 1],
        c=encoded, cmap=cmap,
        alpha=0.5, s=8, linewidths=0
    )

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cmap(i / len(classes)), markersize=7, label=str(cls))
        for i, cls in enumerate(classes)
    ]
    ax.legend(handles=legend_handles, title='Label', bbox_to_anchor=(1.05, 1),
              loc='upper left', fontsize=7)

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
        print("\nRunning UMAP...")
        umap_2d = run_umap(z)
        scatter_plot(umap_2d, actions, 'UMAP — colored by action',
                     os.path.join(args.out_dir, 'umap_actions.png'))
        scatter_plot(umap_2d, games, 'UMAP — colored by game',
                     os.path.join(args.out_dir, 'umap_games.png'))

    print(f"\nDone. Plots saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
