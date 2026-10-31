#!/usr/bin/env python
"""
Layer-number ablation: compare 2 / 3 / 4 ResGAT blocks.

Usage:
    python ablation_study/run_layer_number_ablation.py \
        --graph_dir data/graphs --splits_pkl data/splits.pkl \
        --num_layers 2 3 4 --device cuda:0
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ablation_study.models import ResGATsAblation
from run import load_graphs, load_splits, build_fold_indices, train_one_fold, set_seed


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Layer-number ablation")
    parser.add_argument("--graph_dir", type=str, required=True)
    parser.add_argument("--splits_pkl", type=str, required=True)
    parser.add_argument("--num_layers", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str,
                        default=os.path.join(root, "models", "weights"))
    parser.add_argument("--log_dir", type=str,
                        default=os.path.join(root, "logs"))
    parser.add_argument("--wandb_mode", type=str, default="disabled")
    args = parser.parse_args()

    set_seed(args.seed)
    graphs, names, labels = load_graphs(args.graph_dir)
    folds, id_to_slides = load_splits(args.splits_pkl)

    for nl in args.num_layers:
        print(f"\n{'='*60}")
        print(f"  NUM LAYERS: {nl}")
        print(f"{'='*60}")

        def _model_cls(input_dim, n_classes=args.n_classes, _nl=nl):
            return ResGATsAblation(input_dim, num_layers=_nl,
                                   n_classes=n_classes)

        fold_metrics = []
        for ti in range(len(folds)):
            tr, va, te = build_fold_indices(names, folds, id_to_slides, ti)
            m = train_one_fold(
                [graphs[i] for i in tr], [graphs[i] for i in va],
                [graphs[i] for i in te], fold_idx=ti,
                epochs=args.epochs, lr=args.lr, seed=args.seed,
                device=args.device, save_dir=args.save_dir,
                n_classes=args.n_classes, model_cls=_model_cls,
                wandb_mode=args.wandb_mode, log_dir=args.log_dir,
                wandb_group=f"ablation_layers_{nl}",
            )
            fold_metrics.append(m)

        accs = np.array([m["acc"] for m in fold_metrics])
        print(f"\n{nl} layers | Acc {np.mean(accs):.2f} +/- {np.std(accs):.2f}")


if __name__ == "__main__":
    main()
