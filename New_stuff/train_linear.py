import argparse
import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import glob
import random
from collections import defaultdict
from pathlib import Path


def build_mlp(in_dim, out_dim, n_hidden, hidden_dim=256):
    if n_hidden == 0:
        return nn.Linear(in_dim, out_dim)
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)

def _get_game_name(path, dump_dir, data=None, no_source=False):
    # p2p files store the real game name inside the .pt dict
    if data is not None and 'game_name' in data:
        return str(data['game_name'])
    parts = Path(path).relative_to(dump_dir).parts
    # no-source layout: dump_dir/<game>/...  → parts[0] is game
    # standard layout:  dump_dir/<source>/<game>/... → parts[1] is game
    return parts[0] if no_source else parts[1]


def _extract_actions(data, file_path=''):
    """Return (actions_tensor, action_names_or_None) from a loaded .pt dict.

    Handles three cases:
      - p2p (latent_actions_dump_2): keyboard_labels (N x K multi-hot float tensor)
      - skipped: list of dicts with 'desc' key → encoded as class indices
      - retro: raw numeric tensor
    """
    raw = data.get('actions')

    # p2p keyboard data
    if raw is None and 'keyboard_labels' in data:
        kl = data['keyboard_labels']
        kl = kl if isinstance(kl, torch.Tensor) else torch.tensor(kl)
        kk = data.get('keyboard_keys')
        key_names = [str(k) for k in (kk.tolist() if isinstance(kk, torch.Tensor) else list(kk))] if kk is not None else None

        if kl.ndim == 1:
            kl = kl.unsqueeze(0)
        kl = (kl > 0).float()
        mb = data.get('mouse_buttons')
        if mb is not None:
            mb = mb if isinstance(mb, torch.Tensor) else torch.tensor(mb)
            mouse_cols = torch.zeros((kl.shape[0], 2), dtype=torch.float32)
            mouse_cols[:, 0] = (mb == 0).float()
            mouse_cols[:, 1] = (mb == 1).float()
            kl = torch.cat([kl, mouse_cols], dim=1)
        return kl, key_names

    if raw is None:
        return None, None
    if isinstance(raw, torch.Tensor):
        raw = raw.tolist()
    elif not isinstance(raw, list):
        try:
            raw = list(raw)
        except Exception:
            return None, None
    if not raw:
        return None, None

    if isinstance(raw[0], dict):
        # latent_actions_dump_2 (p2p): dicts with 'action' key
        if 'latent_actions_dump_2' in file_path and 'action' in raw[0]:
            actions = [a['action'] for a in raw]
            try:
                t = torch.as_tensor(actions)
                return (t.unsqueeze(0) if t.ndim == 0 else t), None
            except Exception:
                return None, None
        # skipped dataset: dicts with 'desc'/'description' key
        labels = [a.get('desc', a.get('description', str(sorted(a.items())))) for a in raw]
        unique_labels = sorted(set(labels))
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        return torch.tensor([label_to_idx[l] for l in labels], dtype=torch.long), unique_labels

    # Standard numeric actions
    try:
        t = torch.as_tensor(raw)
        return (t.unsqueeze(0) if t.ndim == 0 else t), None
    except Exception:
        return None, None


def _build_dataset(samples):
    z = torch.stack([sample[0] for sample in samples], dim=0)
    actions = torch.stack([sample[1] for sample in samples], dim=0)
    games = [sample[2] for sample in samples]
    return z, actions, games


def load_data(test_ratio=0.2, seed=42, dataset='both', dump_dir='latent_actions_dump', no_source=False):
    import os as _os
    if no_source:
        files = sorted(glob.glob(_os.path.join(dump_dir, '**', 'latent_actions.pt'), recursive=True))
    elif dataset == 'both':
        files = sorted(glob.glob(_os.path.join(dump_dir, '**', 'latent_actions.pt'), recursive=True))
    else:
        files = sorted(glob.glob(_os.path.join(dump_dir, dataset, '**', 'latent_actions.pt'), recursive=True))
    if not files:
        raise RuntimeError(f'No latent_actions.pt files found under {dump_dir}.')

    samples_by_game = defaultdict(list)
    unique_games = []

    for f in files:
        try:
            data = torch.load(f, map_location='cpu')
        except Exception as e:
            print(f"Skipping {f}: {e}")
            continue

        game_name = _get_game_name(f, dump_dir, data, no_source=no_source)

        actions, _ = _extract_actions(data, file_path=f)
        if actions is None or len(actions) == 0:
            print(f"Skipping {f}: no actions found.")
            continue

        z = torch.as_tensor(data['z_mu'], dtype=torch.float32)
        if z.ndim == 1:
            z = z.unsqueeze(0)

        n = min(z.shape[0], actions.shape[0])
        z = z[:n]
        actions = actions[:n]

        if actions.ndim == 2 and actions.shape[1] == 1 and torch.all(actions == actions.long().to(actions.dtype)):
            actions = actions.squeeze(1)

        if game_name not in samples_by_game:
            unique_games.append(game_name)
        for sample_z, sample_action in zip(z, actions):
            samples_by_game[game_name].append((sample_z, sample_action, game_name))

    min_count = min(len(samples) for samples in samples_by_game.values())
    if min_count < 2:
        raise RuntimeError('Need at least two samples per game to create a train/test split.')

    test_per_game = max(1, int(round(min_count * test_ratio)))
    test_per_game = min(test_per_game, min_count - 1)

    rng = random.Random(seed)
    train_samples = []
    test_samples = []
    for game_name in unique_games:
        game_samples = list(samples_by_game[game_name])
        rng.shuffle(game_samples)
        test_samples.extend(game_samples[:test_per_game])
        train_samples.extend(game_samples[test_per_game:])

    train_z, train_actions, train_games = _build_dataset(train_samples)
    test_z, test_actions, test_games = _build_dataset(test_samples)

    if train_actions.ndim == 1:
        train_actions = train_actions.view(-1)
        test_actions = test_actions.view(-1)
        num_actions = int(torch.max(torch.cat([train_actions, test_actions])).item()) + 1
        action_mode = 'multiclass'
    elif torch.all(train_actions.sum(dim=1) == 1) and torch.all((train_actions == 0) | (train_actions == 1)):
        # one-hot → class indices
        train_actions = train_actions.argmax(dim=1)
        test_actions = test_actions.argmax(dim=1)
        num_actions = int(max(train_actions.max().item(), test_actions.max().item())) + 1
        action_mode = 'multiclass'
    else:
        num_actions = train_actions.shape[1]
        action_mode = 'multilabel'

    game_to_idx = {game_name: idx for idx, game_name in enumerate(unique_games)}
    train_games = torch.tensor([game_to_idx[game_name] for game_name in train_games], dtype=torch.long)
    test_games = torch.tensor([game_to_idx[game_name] for game_name in test_games], dtype=torch.long)

    train_dataset = TensorDataset(train_z, train_actions, train_games)
    test_dataset = TensorDataset(test_z, test_actions, test_games)

    return train_dataset, test_dataset, num_actions, unique_games, action_mode


def _accuracy_from_logits(logits, targets, action_mode):
    if action_mode == 'multiclass':
        predictions = logits.argmax(dim=1)
        return (predictions == targets.long()).float()

    predictions = (torch.sigmoid(logits) >= 0.5).to(targets.dtype)
    return (predictions == targets).view(targets.size(0), -1).float().mean(dim=1)


def evaluate(model, loader, action_mode, unique_games, device):
    model.eval()
    total_correct = 0.0
    total_count = 0
    per_game_correct = {game_name: 0.0 for game_name in unique_games}
    per_game_count = {game_name: 0 for game_name in unique_games}
    per_action_correct = {}
    per_action_count = {}

    with torch.no_grad():
        for batch_z, batch_actions, batch_games in loader:
            batch_z = batch_z.to(device)
            batch_actions = batch_actions.to(device)
            logits = model(batch_z)
            batch_correct = _accuracy_from_logits(logits, batch_actions, action_mode).cpu()
            batch_actions = batch_actions.cpu()
            total_correct += batch_correct.sum().item()
            total_count += batch_correct.numel()

            for game_idx in batch_games.unique(sorted=True):
                game_mask = batch_games == game_idx
                game_name = unique_games[game_idx.item()]
                per_game_correct[game_name] += batch_correct[game_mask].sum().item()
                per_game_count[game_name] += game_mask.sum().item()

            # Track per-action accuracy
            for action_idx in batch_actions.unique(sorted=True):
                action_mask = batch_actions == action_idx
                action_key = action_idx.item() if batch_actions.ndim == 1 else tuple(action_idx.cpu().numpy())
                if action_key not in per_action_correct:
                    per_action_correct[action_key] = 0.0
                    per_action_count[action_key] = 0
                per_action_correct[action_key] += batch_correct[action_mask].sum().item()
                per_action_count[action_key] += action_mask.sum().item()

    total_accuracy = total_correct / total_count if total_count else 0.0
    per_game_accuracy = {
        game_name: (per_game_correct[game_name] / per_game_count[game_name] if per_game_count[game_name] else 0.0)
        for game_name in unique_games
    }
    per_action_accuracy = {
        action_key: (per_action_correct[action_key] / per_action_count[action_key] if per_action_count[action_key] else 0.0)
        for action_key in per_action_correct.keys()
    }
    return total_accuracy, per_game_accuracy, per_action_accuracy


def train_multiclass_model(model, loader, criterion, optimizer, epochs, device, target_index=1, mask=None):
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_z, batch_targets, batch_games in loader:
            batch_z = batch_z.to(device)
            if mask:
                batch_z = batch_z.clone()
                batch_z[:, mask] = 0.0
            if target_index == 1:
                targets = batch_targets.to(device)
            elif target_index == 2:
                targets = batch_games.to(device)
            else:
                raise ValueError(f"Unsupported target_index: {target_index}")

            optimizer.zero_grad()
            logits = model(batch_z)
            loss = criterion(logits, targets.long().view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}")


def evaluate_multiclass_model(model, loader, unique_games, device, target_index=1):
    model.eval()
    total_correct = 0.0
    total_count = 0
    per_game_correct = {game_name: 0.0 for game_name in unique_games}
    per_game_count = {game_name: 0 for game_name in unique_games}

    with torch.no_grad():
        for batch_z, batch_targets, batch_games in loader:
            batch_z = batch_z.to(device)
            if target_index == 1:
                targets = batch_targets.to(device)
            elif target_index == 2:
                targets = batch_games.to(device)
            else:
                raise ValueError(f"Unsupported target_index: {target_index}")

            logits = model(batch_z)
            predictions = logits.argmax(dim=1)
            batch_correct = (predictions == targets.long().view(-1)).float().cpu()
            total_correct += batch_correct.sum().item()
            total_count += batch_correct.numel()

            for game_idx in batch_games.unique(sorted=True):
                game_mask = batch_games == game_idx
                game_name = unique_games[game_idx.item()]
                per_game_correct[game_name] += batch_correct[game_mask].sum().item()
                per_game_count[game_name] += game_mask.sum().item()

    total_accuracy = total_correct / total_count if total_count else 0.0
    per_game_accuracy = {
        game_name: (per_game_correct[game_name] / per_game_count[game_name] if per_game_count[game_name] else 0.0)
        for game_name in unique_games
    }
    return total_accuracy, per_game_accuracy

def load_data_per_game(test_ratio=0.2, seed=42, dataset='both', dump_dir='latent_actions_dump', no_source=False):
    """Load data grouped by game. Returns dict: game_name -> (train_dataset, test_dataset, num_actions, action_mode)."""
    import os as _os
    if no_source:
        files = sorted(glob.glob(_os.path.join(dump_dir, '**', 'latent_actions.pt'), recursive=True))
    elif dataset == 'both':
        files = sorted(glob.glob(_os.path.join(dump_dir, '**', 'latent_actions.pt'), recursive=True))
    else:
        files = sorted(glob.glob(_os.path.join(dump_dir, dataset, '**', 'latent_actions.pt'), recursive=True))
    if not files:
        raise RuntimeError(f'No latent_actions.pt files found under {dump_dir}.')

    samples_by_game = defaultdict(list)

    for f in files:
        try:
            data = torch.load(f, map_location='cpu')
        except Exception as e:
            print(f"Skipping {f}: {e}")
            continue

        game_name = _get_game_name(f, dump_dir, data, no_source=no_source)

        actions, _ = _extract_actions(data, file_path=f)
        if actions is None or len(actions) == 0:
            print(f"Skipping {f}: no actions found.")
            continue

        z = torch.as_tensor(data['z_mu'], dtype=torch.float32)
        if z.ndim == 1:
            z = z.unsqueeze(0)

        n = min(z.shape[0], actions.shape[0])
        z = z[:n]
        actions = actions[:n]

        if actions.ndim == 2 and actions.shape[1] == 1 and torch.all(actions == actions.long().to(actions.dtype)):
            actions = actions.squeeze(1)

        for sample_z, sample_action in zip(z, actions):
            samples_by_game[game_name].append((sample_z, sample_action))

    rng = random.Random(seed)
    game_datasets = {}

    for game_name, samples in samples_by_game.items():
        if len(samples) < 2:
            print(f"Skipping {game_name}: only {len(samples)} sample(s).")
            continue

        game_samples = list(samples)
        rng.shuffle(game_samples)

        test_count = max(1, int(round(len(game_samples) * test_ratio)))
        test_count = min(test_count, len(game_samples) - 1)

        test_s = game_samples[:test_count]
        train_s = game_samples[test_count:]

        train_z = torch.stack([s[0] for s in train_s])
        test_z = torch.stack([s[0] for s in test_s])
        train_actions = torch.stack([s[1] for s in train_s])
        test_actions = torch.stack([s[1] for s in test_s])

        if train_actions.ndim == 1:
            num_actions = int(torch.max(torch.cat([train_actions, test_actions])).item()) + 1
            action_mode = 'multiclass'
        elif torch.all(train_actions.sum(dim=1) == 1) and torch.all((train_actions == 0) | (train_actions == 1)):
            train_actions = train_actions.argmax(dim=1)
            test_actions = test_actions.argmax(dim=1)
            num_actions = int(max(train_actions.max().item(), test_actions.max().item())) + 1
            action_mode = 'multiclass'
        else:
            num_actions = train_actions.shape[1]
            action_mode = 'multilabel'

        game_datasets[game_name] = (
            TensorDataset(train_z, train_actions),
            TensorDataset(test_z, test_actions),
            num_actions,
            action_mode,
        )

    return game_datasets


def train_per_game(game_datasets, args, device):
    """Train and evaluate a separate model for each game. Returns dict: game_name -> test_accuracy."""
    results = {}

    for game_name, (train_dataset, test_dataset, num_actions, action_mode) in game_datasets.items():
        print(f"\n{'='*60}")
        print(f"Game: {game_name}  |  train={len(train_dataset)}  test={len(test_dataset)}  actions={num_actions}  mode={action_mode}")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        in_dim = train_dataset.tensors[0].shape[1]
        model = build_mlp(in_dim, num_actions, args.action_hidden_layers).to(device)
        criterion = nn.CrossEntropyLoss() if action_mode == 'multiclass' else nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            for batch_z, batch_actions in train_loader:
                batch_z = batch_z.to(device)
                if args.mask:
                    batch_z = batch_z.clone()
                    batch_z[:, args.mask] = 0.0
                batch_actions = batch_actions.to(device)
                optimizer.zero_grad()
                pred = model(batch_z)
                if action_mode == 'multiclass':
                    loss = criterion(pred, batch_actions.long().view(-1))
                else:
                    loss = criterion(pred, batch_actions.float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{args.epochs}, Loss: {total_loss / len(train_loader):.4f}")

        model.eval()
        total_correct = 0.0
        total_count = 0
        # Multilabel (p2p) accumulators
        exact_match_correct = 0.0
        hamming_wrong = 0.0
        hamming_total = 0
        per_key_tp = torch.zeros(num_actions) if action_mode == 'multilabel' else None
        per_key_fp = torch.zeros(num_actions) if action_mode == 'multilabel' else None
        per_key_fn = torch.zeros(num_actions) if action_mode == 'multilabel' else None
        baseline_correct = 0.0

        with torch.no_grad():
            for batch_z, batch_actions in test_loader:
                batch_z = batch_z.to(device)
                batch_actions = batch_actions.to(device)
                logits = model(batch_z)
                if action_mode == 'multiclass':
                    batch_correct = (logits.argmax(dim=1) == batch_actions.long().view(-1)).float()
                else:
                    preds = (torch.sigmoid(logits) >= 0.5).to(batch_actions.dtype)
                    match = (preds == batch_actions)
                    batch_correct = match.view(batch_actions.size(0), -1).float().mean(dim=1)
                    # Exact match
                    exact_match_correct += match.all(dim=1).float().sum().item()
                    # Hamming
                    hamming_wrong += (~match).float().sum().item()
                    hamming_total += batch_actions.numel()
                    # Per-key F1
                    preds_cpu = preds.cpu()
                    targets_cpu = batch_actions.cpu()
                    per_key_tp += ((preds_cpu == 1) & (targets_cpu == 1)).float().sum(dim=0)
                    per_key_fp += ((preds_cpu == 1) & (targets_cpu == 0)).float().sum(dim=0)
                    per_key_fn += ((preds_cpu == 0) & (targets_cpu == 1)).float().sum(dim=0)
                    # Baseline: always predict 0
                    baseline_correct += (batch_actions == 0).float().sum().item()
                total_correct += batch_correct.sum().item()
                total_count += batch_correct.numel()

        accuracy = total_correct / total_count if total_count else 0.0
        result = {
            'accuracy': accuracy,
            'n_train': len(train_dataset),
            'n_test': len(test_dataset),
            'n_actions': num_actions,
            'action_mode': action_mode,
        }

        if action_mode == 'multilabel':
            exact_match_acc = exact_match_correct / total_count if total_count else 0.0
            hamming_loss = hamming_wrong / hamming_total if hamming_total else 0.0
            f1_scores = []
            for k in range(num_actions):
                denom = 2 * per_key_tp[k] + per_key_fp[k] + per_key_fn[k]
                f1_scores.append((2 * per_key_tp[k] / denom).item() if denom > 0 else 0.0)
            result['exact_match_acc'] = exact_match_acc
            result['hamming_loss'] = hamming_loss
            result['per_key_f1'] = f1_scores
            baseline = baseline_correct / hamming_total if hamming_total else 0.0
            print(f"  Test accuracy (element-wise): {accuracy:.4f}")
            print(f"  Exact match accuracy:         {exact_match_acc:.4f}")
            print(f"  Hamming loss:                 {hamming_loss:.4f}")
            print(f"  Baseline (always predict 0):  {baseline:.4f}")
            print(f"  Per-key F1: {[f'{v:.3f}' for v in f1_scores]}")
        else:
            print(f"  Test accuracy: {accuracy:.4f}")

        results[game_name] = result

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--action_hidden_layers', type=int, default=1)
    parser.add_argument('--game_hidden_layers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--mask', type=int, nargs='*', default=[],
                        help='Dimensions to zero out during training (e.g. --mask 0 3 5)')
    parser.add_argument('--dataset', type=str, default='both', choices=['adaworld', 'olafworld', 'both'],
                        help='Which dataset to train on (default: both). Ignored when --no-source is set.')
    parser.add_argument('--dump-dir', type=str, default='latent_actions_dump',
                        help='Root of the latent_actions_dump folder (default: ./latent_actions_dump)')
    parser.add_argument('--out-csv', type=str, default=None,
                        help='Path to save per-game results as CSV (e.g. ./results/per_game.csv)')
    parser.add_argument('--per_game', action='store_true',
                        help='Train a separate model for each game instead of a single shared model')
    parser.add_argument('--no-source', action='store_true',
                        help='Data has no source subfolder: dump-dir/<game>/... instead of dump-dir/<source>/<game>/...')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.per_game:
        print("Loading per-game data...")
        game_datasets = load_data_per_game(dataset=args.dataset, dump_dir=args.dump_dir,
                                           no_source=args.no_source)
        print(f"Games: {list(game_datasets.keys())}")
        per_game_results = train_per_game(game_datasets, args, device)
        print(f"\n{'='*60}")
        print("Per-game accuracy summary (sorted by accuracy):")
        sorted_results = sorted(per_game_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        for game_name, info in sorted_results:
            print(f"  {game_name}: {info['accuracy']:.4f}  (train={info['n_train']}, test={info['n_test']}, actions={info['n_actions']})")
        accs = [info['accuracy'] for info in per_game_results.values()]
        print(f"  Mean: {sum(accs)/len(accs):.4f}" if accs else "  No results.")

        if args.out_csv:
            os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
            fieldnames = ['game', 'accuracy', 'exact_match_acc', 'hamming_loss', 'per_key_f1',
                          'n_train', 'n_test', 'n_actions', 'action_mode']
            with open(args.out_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                for game_name, info in sorted_results:
                    row = {'game': game_name, **info}
                    if 'per_key_f1' in row:
                        row['per_key_f1'] = str([round(v, 4) for v in row['per_key_f1']])
                    writer.writerow(row)
            print(f"\nResults saved to {args.out_csv}")
        return

    print("Loading data...")
    train_dataset, test_dataset, num_actions, unique_games, action_mode = load_data(dataset=args.dataset, dump_dir=args.dump_dir, no_source=args.no_source)
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print(f"Games: {unique_games}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print('Dataloader sizes:', len(train_loader), len(test_loader))
    print('Masking dimensions:', args.mask)

    in_dim = train_dataset.tensors[0].shape[1]
    epochs = args.epochs

    model = build_mlp(in_dim, num_actions, args.action_hidden_layers).to(device)
    criterion = nn.CrossEntropyLoss() if action_mode == 'multiclass' else nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Training started...")
    for epoch in range(epochs):
        total_loss = 0
        for batch_z, batch_actions, _ in train_loader:
            batch_z = batch_z.to(device)
            if args.mask:
                batch_z = batch_z.clone()
                batch_z[:, args.mask] = 0.0
            batch_actions = batch_actions.to(device)
            optimizer.zero_grad()
            pred = model(batch_z)
            if action_mode == 'multiclass':
                loss = criterion(pred, batch_actions.long().view(-1))
            else:
                loss = criterion(pred, batch_actions.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    print("Testing started...")
    total_accuracy, per_game_accuracy, per_action_accuracy = evaluate(model, test_loader, action_mode, unique_games, device)
    print(f"Total accuracy: {total_accuracy:.4f}")
    print("\nPer-action accuracy:")
    for action_key in sorted(per_action_accuracy.keys()):
        print(f"  A{action_key}: {per_action_accuracy[action_key]:.4f}")
    # for game_name in unique_games:
    #     print(f"{game_name}: {per_game_accuracy[game_name]:.4f}")

    print("Training game predictor...")
    game_model = build_mlp(in_dim, len(unique_games), args.game_hidden_layers).to(device)
    game_criterion = nn.CrossEntropyLoss()
    game_optimizer = optim.Adam(game_model.parameters(), lr=1e-3)
    train_multiclass_model(game_model, train_loader, game_criterion, game_optimizer, epochs, device, target_index=2, mask=args.mask)

    print("Testing game predictor...")
    game_accuracy, per_game_game_accuracy = evaluate_multiclass_model(game_model, test_loader, unique_games, device, target_index=2)
    print(f"Game accuracy: {game_accuracy:.4f}")
    # for game_name in unique_games:
    #     print(f"{game_name}: {per_game_game_accuracy[game_name]:.4f}")

if __name__ == '__main__':
    main()
