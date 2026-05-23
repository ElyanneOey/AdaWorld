"""Print action label counts for a latent_actions dump directory."""
import argparse
import glob
import os
from collections import Counter
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dump-dir', type=str, required=True)
    p.add_argument('--source', type=str, default=None,
                   help='Only load from this source subfolder, e.g. adaworld or olafworld')
    p.add_argument('--single-action', action='store_true',
                   help='Only count samples with a single key pressed (no + combinations)')
    return p.parse_args()


def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.dump_dir, '**', 'latent_actions.pt'), recursive=True))
    if not files:
        raise RuntimeError(f'No latent_actions.pt files found under {args.dump_dir}')

    if args.source:
        files = [f for f in files if Path(f).relative_to(args.dump_dir).parts[0] == args.source]

    print(f'Found {len(files)} files')

    counter = Counter()
    games = set()
    skipped = 0

    for f in files:
        try:
            data = torch.load(f, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f'  Skipping {f}: {e}')
            skipped += 1
            continue

        if 'game_name' in data:
            games.add(str(data['game_name']))

        kl = data.get('keyboard_labels')
        if kl is None:
            continue
        kl = torch.as_tensor(kl)
        if kl.ndim == 1:
            kl = kl.unsqueeze(0)
        kl = (kl > 0)

        kk = data.get('keyboard_keys')
        key_names = [str(k) for k in (kk.tolist() if isinstance(kk, torch.Tensor) else list(kk))] if kk is not None else None

        mb = data.get('mouse_buttons')
        if mb is not None:
            mb = torch.as_tensor(mb)

        for i in range(kl.shape[0]):
            pressed = [key_names[j] if key_names else str(j) for j, v in enumerate(kl[i].tolist()) if v]
            if mb is not None and i < len(mb):
                mb_val = mb[i].item()
                if mb_val == 0:
                    pressed.append('left_click')
                elif mb_val == 1:
                    pressed.append('right_click')
            label = '+'.join(sorted(pressed)) if pressed else 'none'
            counter[label] += 1

    if args.single_action:
        counter = Counter({k: v for k, v in counter.items() if '+' not in k})

    total = sum(counter.values())
    print(f'Games: {sorted(games)}')
    print(f'Skipped files: {skipped}')
    print(f'\nAction labels ({len(counter)}) — total samples: {total}')
    for label, count in sorted(counter.items(), key=lambda x: -x[1]):
        print(f'  {label}: {count}')


if __name__ == '__main__':
    main()
