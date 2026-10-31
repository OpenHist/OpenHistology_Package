#!/usr/bin/env python
"""
Cross-site domain adaptation for ResGAT.

Three-phase protocol:
  1. Pre-train on source-domain graphs (e.g. Wake Forest).
  2. Zero-shot evaluation on target-domain test set (e.g. Stanford).
  3. Few-shot fine-tuning on a small labelled target-domain subset,
     freezing everything except GraphNorm + classifier layers.

Usage:
    python domain_adaptation/run_cross_site.py \
        --graph_dir data/ac/graphs \
        --source_prefix WF --target_prefix S \
        --few_shot_samples 3 6 9 --device cuda:0
"""

import argparse
import copy
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                              f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ResGATs
from utils.metrics import collect_predictions
from utils.logger import TrainLogger

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_graphs(graph_dir: str):
    from run import load_graphs as _lg
    return _lg(graph_dir)


# ------------------------------------------------------------------
# Domain splitting
# ------------------------------------------------------------------

def split_by_prefix(names, graphs, labels, source_prefix, target_prefix):
    src_idx = [i for i, n in enumerate(names)
               if n.startswith(source_prefix)]
    tgt_idx = [i for i, n in enumerate(names)
               if n.startswith(target_prefix)]
    return src_idx, tgt_idx


def make_source_splits(graphs, labels, indices, seed=42,
                       train_ratio=0.7, val_ratio=0.15):
    set_seed(seed)
    n = len(indices)
    labs = [labels[i] for i in indices]
    tv_idx, te_idx = train_test_split(
        list(range(n)), test_size=1 - train_ratio - val_ratio,
        stratify=labs, random_state=seed)
    tv_labs = [labs[i] for i in tv_idx]
    tr_idx, va_idx = train_test_split(
        tv_idx, test_size=val_ratio / (train_ratio + val_ratio),
        stratify=tv_labs, random_state=seed)
    return ([indices[i] for i in tr_idx],
            [indices[i] for i in va_idx],
            [indices[i] for i in te_idx])


def make_target_few_shot(graphs, labels, indices, n_shots, seed=42,
                         test_size=12, val_size=3):
    set_seed(seed)
    labs = [labels[i] for i in indices]
    _, te_local = train_test_split(
        list(range(len(indices))), test_size=test_size,
        stratify=labs, random_state=seed)
    remaining = [i for i in range(len(indices)) if i not in te_local]
    random.shuffle(remaining)
    tr_local = remaining[:n_shots]
    va_local = remaining[n_shots:n_shots + val_size]
    return ([indices[i] for i in tr_local],
            [indices[i] for i in va_local],
            [indices[i] for i in te_local])


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate_model(model, loader, device, criterion=None):
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, out = model(batch)
            if criterion:
                total_loss += criterion(out, batch.y).item()
            probs = torch.exp(out)
            all_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    preds, labs = np.array(all_preds), np.array(all_labels)
    metrics = {
        "accuracy": balanced_accuracy_score(labs, preds),
        "f1": f1_score(labs, preds, average="weighted", zero_division=0),
    }
    cm = confusion_matrix(labs, preds)
    if cm.shape[0] >= 2:
        metrics["class_0_acc"] = cm[0, 0] / max(cm[0].sum(), 1)
        metrics["class_1_acc"] = cm[1, 1] / max(cm[1].sum(), 1)
    try:
        metrics["auc"] = roc_auc_score(labs, all_probs)
    except Exception:
        metrics["auc"] = 0.5
    avg_loss = total_loss / max(len(loader), 1) if criterion else 0.0
    return avg_loss, metrics


# ------------------------------------------------------------------
# Freezing strategies
# ------------------------------------------------------------------

def freeze_except_graphnorm_fc(model):
    for p in model.parameters():
        p.requires_grad = False
    for name, mod in model.named_modules():
        if "GraphNorm" in type(mod).__name__ or "graph_norm" in name.lower():
            for p in mod.parameters():
                p.requires_grad = True
    for p in model.fc1.parameters():
        p.requires_grad = True
    for p in model.fc2.parameters():
        p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable}/{total} ({trainable / total * 100:.1f}%)")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(
        description="Cross-site domain adaptation for ResGAT.")
    parser.add_argument("--graph_dir", type=str, required=True)
    parser.add_argument("--source_prefix", type=str, default="WF",
                        help="Name prefix for source-domain slides.")
    parser.add_argument("--target_prefix", type=str, default="S",
                        help="Name prefix for target-domain slides.")
    parser.add_argument("--few_shot_samples", nargs="+", type=int,
                        default=[3, 6, 9])
    parser.add_argument("--source_epochs", type=int, default=20)
    parser.add_argument("--ft_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ft_lr", type=float, default=3e-4)
    parser.add_argument("--freeze_strategy", type=str, default="graphnorm_fc",
                        choices=["graphnorm_fc", "none"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str,
                        default=os.path.join(root, "models", "weights"))
    parser.add_argument("--log_dir", type=str,
                        default=os.path.join(root, "logs"))
    parser.add_argument("--wandb_mode", type=str, default="disabled")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    logger = TrainLogger(args.log_dir, "domain_adaptation") if args.log_dir else None

    graphs, names, labels = load_graphs(args.graph_dir)
    src_idx, tgt_idx = split_by_prefix(names, graphs, labels,
                                        args.source_prefix,
                                        args.target_prefix)
    print(f"Source ({args.source_prefix}): {len(src_idx)} | "
          f"Target ({args.target_prefix}): {len(tgt_idx)}")

    # --- Phase 1: Source pre-training ---
    print(f"\n{'='*60}\n  PHASE 1: Source Pre-training\n{'='*60}")
    tr, va, te = make_source_splits(graphs, labels, src_idx, seed=args.seed)
    in_dim = graphs[tr[0]].x.size(1)

    model = ResGATs(in_dim).to(device)
    train_loader = DataLoader([graphs[i] for i in tr], batch_size=1,
                              shuffle=True)
    val_loader = DataLoader([graphs[i] for i in va], batch_size=1)
    test_loader = DataLoader([graphs[i] for i in te], batch_size=1)

    train_labels = np.array([labels[i] for i in tr])
    cw = compute_class_weight("balanced", classes=np.arange(2),
                              y=train_labels)
    criterion = nn.NLLLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-4)

    best_wts, best_val = None, math.inf
    for epoch in range(args.source_epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            _, out = model(batch)
            loss = criterion(out, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tl, tm = evaluate_model(model, train_loader, device, criterion)
        vl, vm = evaluate_model(model, val_loader, device, criterion)
        if logger:
            logger.log_epoch(epoch + 1, tl, tm["accuracy"] * 100,
                             vl, vm["accuracy"] * 100, phase="source")
        if vl < best_val:
            best_val = vl
            best_wts = copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch + 1} | val_loss {vl:.4f} | "
              f"val_bacc {vm['accuracy']:.4f}")

    model.load_state_dict(best_wts)
    torch.save(best_wts, os.path.join(args.save_dir, "source_best.pth"))
    _, src_test = evaluate_model(model, test_loader, device)
    if logger:
        logger.log_test(phase="source", bacc=src_test["accuracy"],
                        auc=src_test["auc"], f1=src_test["f1"])
    print(f"Source test BAcc: {src_test['accuracy']:.4f} | "
          f"AUC: {src_test['auc']:.4f}")

    # --- Phase 2: Zero-shot target evaluation ---
    print(f"\n{'='*60}\n  PHASE 2: Zero-shot Target Evaluation\n{'='*60}")
    _, _, tgt_te = make_target_few_shot(graphs, labels, tgt_idx, n_shots=0,
                                        seed=args.seed)
    tgt_test_loader = DataLoader([graphs[i] for i in tgt_te], batch_size=1)
    _, zs_metrics = evaluate_model(model, tgt_test_loader, device)
    if logger:
        logger.log_test(phase="zero_shot", bacc=zs_metrics["accuracy"],
                        auc=zs_metrics["auc"], f1=zs_metrics["f1"])
    print(f"Zero-shot target BAcc: {zs_metrics['accuracy']:.4f} | "
          f"AUC: {zs_metrics['auc']:.4f}")

    # --- Phase 3: Few-shot fine-tuning ---
    for n_shots in args.few_shot_samples:
        print(f"\n{'='*60}\n  PHASE 3: {n_shots}-shot Fine-tuning\n{'='*60}")
        ft_tr, ft_va, ft_te = make_target_few_shot(
            graphs, labels, tgt_idx, n_shots=n_shots, seed=args.seed)

        ft_model = copy.deepcopy(model)
        if args.freeze_strategy == "graphnorm_fc":
            freeze_except_graphnorm_fc(ft_model)
        ft_opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, ft_model.parameters()),
            lr=args.ft_lr, weight_decay=5e-4)
        ft_crit = nn.NLLLoss()

        ft_train_loader = DataLoader([graphs[i] for i in ft_tr],
                                     batch_size=1, shuffle=True)
        ft_val_loader = DataLoader([graphs[i] for i in ft_va], batch_size=1)
        ft_test_loader = DataLoader([graphs[i] for i in ft_te], batch_size=1)

        best_ft, best_ft_loss = None, math.inf
        for epoch in range(args.ft_epochs):
            ft_model.train()
            for batch in ft_train_loader:
                batch = batch.to(device)
                _, out = ft_model(batch)
                loss = ft_crit(out, batch.y)
                ft_opt.zero_grad()
                loss.backward()
                ft_opt.step()
            vl, _ = evaluate_model(ft_model, ft_val_loader, device, ft_crit)
            if vl < best_ft_loss:
                best_ft_loss = vl
                best_ft = copy.deepcopy(ft_model.state_dict())

        ft_model.load_state_dict(best_ft)
        _, tgt_m = evaluate_model(ft_model, ft_test_loader, device)
        _, src_m = evaluate_model(ft_model, test_loader, device)

        fwt = tgt_m["accuracy"] - zs_metrics["accuracy"]
        bwt = src_m["accuracy"] - src_test["accuracy"]

        if logger:
            logger.log_test(phase=f"few_shot_{n_shots}",
                            target_bacc=tgt_m["accuracy"],
                            source_bacc=src_m["accuracy"],
                            fwt=fwt, bwt=bwt,
                            target_auc=tgt_m["auc"],
                            target_f1=tgt_m["f1"])

        print(f"  Target BAcc: {zs_metrics['accuracy']:.4f} -> "
              f"{tgt_m['accuracy']:.4f} (FWT: {fwt:+.4f})")
        print(f"  Source BAcc: {src_test['accuracy']:.4f} -> "
              f"{src_m['accuracy']:.4f} (BWT: {bwt:+.4f})")

        torch.save(best_ft, os.path.join(
            args.save_dir, f"ft_{n_shots}shot_best.pth"))

    if logger:
        logger.close()
    print("\nDomain adaptation experiment complete.")


if __name__ == "__main__":
    main()
