#!/usr/bin/env python
"""
Unified training script for ResGAT on multiple WSI datasets.

Supports: AC (appendiceal cancer), TCGA-NSCLC, TCGA-ESCA, BRACS.

Usage examples
--------------
# AC dataset (binary, patient-level 5-fold)
python run.py --dataset ac --graph_dir data/ac/graphs \\
    --splits_pkl data/ac/AC_five_fold_splits.pkl --epochs 20

# TCGA-NSCLC (binary, case-level 5-fold)
python run.py --dataset tcga_nsclc --graph_dir data/tcga_nsclc/graphs \\
    --splits_pkl data/tcga_nsclc/splits.pkl --epochs 20

# BRACS (7-class, slide-level 5-fold)
python run.py --dataset bracs --graph_dir data/bracs/graphs \\
    --splits_pkl data/bracs/splits.pkl --n_classes 7 --epochs 50
"""

import argparse
import copy
import math
import os
import pickle
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                              f1_score, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from models import ResGATs
from utils.metrics import collect_predictions, evaluate_metrics
from utils.logger import TrainLogger

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------
# Graph loading
# ------------------------------------------------------------------

def load_graphs(graph_dir: str) -> Tuple[List[Data], List[str], List[int]]:
    """Load ``Weighted_Graph_<id>-<label>.pt`` files from *graph_dir*."""
    all_files = sorted(
        os.path.join(graph_dir, f)
        for f in os.listdir(graph_dir)
        if f.endswith(".pt") and f.startswith("Weighted_Graph_")
    )
    graphs, names, labels = [], [], []
    for path in all_files:
        stem = os.path.splitext(os.path.basename(path))[0]
        core = stem.replace("Weighted_Graph_", "", 1)
        try:
            slide_id, lbl_str = core.rsplit("-", 1)
            lbl = int(lbl_str)
        except Exception:
            slide_id, lbl = core, None

        data: Data = torch.load(path, map_location="cpu", weights_only=False)
        if lbl is None:
            lbl = int(data.y.view(-1)[0].item()) if hasattr(data, "y") and data.y.numel() > 0 else 0
        else:
            data.y = torch.tensor(lbl, dtype=torch.long)
        graphs.append(data)
        names.append(slide_id)
        labels.append(lbl)
    print(f"Loaded {len(graphs)} graphs from {graph_dir}")
    return graphs, names, labels


# ------------------------------------------------------------------
# Split loading helpers (patient-level / case-level / slide-level)
# ------------------------------------------------------------------

def load_splits(pickle_path: str):
    """Load cross-validation splits from a pickle file.

    Expected keys: ``folds`` (list of id-lists), plus one of
    ``patient_slides`` / ``case_slides`` / ``slide_ids``, and
    ``patient_labels`` / ``case_labels`` / ``slide_labels``.
    """
    with open(pickle_path, "rb") as f:
        obj = pickle.load(f)
    folds = obj["folds"]
    id_to_slides = (obj.get("patient_slides") or obj.get("case_slides")
                    or obj.get("slide_ids") or {})
    return folds, id_to_slides


def build_fold_indices(names: List[str], folds, id_to_slides: dict,
                       test_fold_idx: int):
    """Map fold membership to graph indices (train / val / test)."""
    base_to_idx: Dict[str, int] = {}
    for idx, n in enumerate(names):
        base_to_idx[str(n)] = idx
        base_to_idx[str(n).split("_")[0]] = idx
        base_to_idx[str(n).split(".")[0]] = idx

    num_folds = len(folds)
    val_fold_idx = (test_fold_idx + 1) % num_folds

    def _collect(id_list):
        idxs = []
        for pid in id_list:
            sids = id_to_slides.get(pid, [pid])
            if not isinstance(sids, list):
                sids = [sids]
            for sid in sids:
                for key in [str(sid), str(sid).split("_")[0],
                            str(sid).split(".")[0]]:
                    if key in base_to_idx:
                        idxs.append(base_to_idx[key])
                        break
        return list(dict.fromkeys(idxs))

    train_ids = [p for i, fold in enumerate(folds)
                 if i != test_fold_idx and i != val_fold_idx for p in fold]
    return (_collect(train_ids), _collect(folds[val_fold_idx]),
            _collect(folds[test_fold_idx]))


# ------------------------------------------------------------------
# Single-fold training
# ------------------------------------------------------------------

def train_one_fold(train_graphs, val_graphs, test_graphs, fold_idx, *,
                   batch_size=1, epochs=70, lr=1e-4, weight_decay=1e-4,
                   seed=42, save_dir=None, device="cuda:0",
                   wandb_project="ResGAT", wandb_group="ResGAT",
                   wandb_mode="disabled", n_classes=2, model_cls=ResGATs,
                   log_dir=None):
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    in_dim = train_graphs[0].x.size(1)
    model = model_cls(in_dim, n_classes=n_classes).to(device)

    train_labels_np = np.array([int(g.y.item()) for g in train_graphs])
    classes = np.arange(n_classes)
    present = np.intersect1d(classes, np.unique(train_labels_np))
    cw = compute_class_weight("balanced", classes=present, y=train_labels_np)
    class_weights = torch.ones(n_classes, dtype=torch.float32)
    for c, w in zip(present, cw):
        class_weights[c] = w
    class_weights = class_weights.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    criterion = nn.NLLLoss(weight=class_weights)

    run_name = f"ResGAT_fold{fold_idx}"
    logger = TrainLogger(log_dir, run_name) if log_dir else None

    run = None
    if HAS_WANDB and wandb_mode != "disabled":
        os.environ["WANDB_MODE"] = wandb_mode
        run = wandb.init(project=wandb_project, group=wandb_group,
                         name=run_name,
                         config={"epochs": epochs, "lr": lr,
                                 "weight_decay": weight_decay,
                                 "batch_size": batch_size})

    best_wts, min_val_loss = None, math.inf
    for epoch in range(epochs):
        # --- train ---
        model.train()
        t_loss, t_outs, t_labs = 0.0, [], []
        for batch in train_loader:
            batch = batch.to(device)
            logits, out = model(batch)
            loss = criterion(out, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            t_outs.append(out.detach())
            t_labs.append(batch.y)

        t_outs = torch.cat(t_outs)
        t_labs = torch.cat(t_labs)
        t_preds, t_true, _, t_probs = collect_predictions(t_outs, t_labs)
        t_loss /= max(1, len(train_loader))
        t_acc, t_auc, t_f1 = evaluate_metrics(t_preds, t_true, t_probs)

        # --- val ---
        model.eval()
        v_loss, v_outs, v_labs = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits, out = model(batch)
                loss = criterion(out, batch.y)
                v_loss += loss.item()
                v_outs.append(out)
                v_labs.append(batch.y)

        v_outs = torch.cat(v_outs)
        v_labs = torch.cat(v_labs)
        v_preds, v_true, _, v_probs = collect_predictions(v_outs, v_labs)
        v_loss /= max(1, len(val_loader))
        v_acc, v_auc, v_f1 = evaluate_metrics(v_preds, v_true, v_probs)

        if run is not None:
            wandb.log({"epoch": epoch + 1,
                        "train/loss": t_loss, "train/acc": t_acc,
                        "val/loss": v_loss, "val/acc": v_acc})

        if logger:
            logger.log_epoch(epoch + 1, t_loss, t_acc, v_loss, v_acc)

        print(f"Epoch {epoch + 1:03d} | "
              f"Train Loss {t_loss:.4f} Acc {t_acc:.2f}% | "
              f"Val Loss {v_loss:.4f} Acc {v_acc:.2f}%")

        if v_loss <= min_val_loss:
            min_val_loss = v_loss
            best_wts = copy.deepcopy(model.state_dict())
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                torch.save(best_wts,
                           os.path.join(save_dir, f"{run_name}_best.pth"))

    # --- test ---
    if best_wts is not None:
        model.load_state_dict(best_wts)
    model.eval()
    te_outs, te_labs = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            _, out = model(batch)
            te_outs.append(out)
            te_labs.append(batch.y)

    te_outs = torch.cat(te_outs)
    te_labs = torch.cat(te_labs)
    te_preds = te_outs.argmax(dim=1).cpu().numpy()
    te_true = te_labs.cpu().numpy()
    te_probs = torch.softmax(te_outs, dim=1).cpu().numpy()

    te_acc = float((te_preds == te_true).mean() * 100)
    te_bacc = balanced_accuracy_score(te_true, te_preds)
    y_score = te_probs[:, 1] if te_probs.shape[1] == 2 else te_probs
    try:
        te_auc = roc_auc_score(
            te_true, y_score,
            multi_class="ovr" if te_probs.shape[1] > 2 else "raise")
    except Exception:
        te_auc = float("nan")
    te_f1 = f1_score(te_true, te_preds,
                     average="macro" if n_classes > 2 else "binary",
                     zero_division=0)

    if run is not None:
        wandb.log({"test/acc": te_acc, "test/bacc": te_bacc,
                    "test/auc": te_auc, "test/f1": te_f1})
        wandb.finish()

    if logger:
        logger.log_test(acc=te_acc, bacc=te_bacc, auc=te_auc, f1=te_f1)
        logger.close()

    print(f"===== Test | Acc {te_acc:.2f}% | BAcc {te_bacc:.4f} | "
          f"AUC {te_auc:.4f} | F1 {te_f1:.4f} =====")
    return {"acc": te_acc, "bacc": te_bacc, "auc": te_auc, "f1": te_f1}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Train ResGAT with k-fold cross-validation.")
    parser.add_argument("--dataset", type=str, default="ac",
                        choices=["ac", "tcga_nsclc", "tcga_esca", "bracs"])
    parser.add_argument("--graph_dir", type=str, required=True)
    parser.add_argument("--splits_pkl", type=str, required=True)
    parser.add_argument("--n_classes", type=int, default=None,
                        help="Number of classes (auto-detected if omitted).")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str,
                        default=os.path.join(root, "models", "weights"))
    parser.add_argument("--log_dir", type=str,
                        default=os.path.join(root, "logs"))
    parser.add_argument("--wandb_mode", type=str, default="disabled",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_project", type=str, default="ResGAT")
    parser.add_argument("--single_fold", type=int, default=None,
                        help="Run only this fold index (0-based).")
    args = parser.parse_args()

    n_classes_map = {"ac": 2, "tcga_nsclc": 2, "tcga_esca": 2, "bracs": 7}
    n_classes = args.n_classes or n_classes_map.get(args.dataset, 2)

    set_seed(args.seed)
    graphs, names, labels = load_graphs(args.graph_dir)
    folds, id_to_slides = load_splits(args.splits_pkl)

    fold_indices = ([args.single_fold] if args.single_fold is not None
                    else list(range(len(folds))))

    all_metrics: List[Dict] = []
    for ti in fold_indices:
        tr_idx, va_idx, te_idx = build_fold_indices(names, folds,
                                                     id_to_slides, ti)
        print(f"\nFold {ti + 1}/{len(folds)} | "
              f"train {len(tr_idx)} | val {len(va_idx)} | test {len(te_idx)}")
        metrics = train_one_fold(
            [graphs[i] for i in tr_idx],
            [graphs[i] for i in va_idx],
            [graphs[i] for i in te_idx],
            fold_idx=ti, batch_size=args.batch_size, epochs=args.epochs,
            lr=args.lr, weight_decay=args.weight_decay, seed=args.seed,
            save_dir=args.save_dir, device=args.device,
            wandb_project=args.wandb_project, wandb_mode=args.wandb_mode,
            n_classes=n_classes, log_dir=args.log_dir,
        )
        all_metrics.append(metrics)

    if all_metrics:
        for key in ["acc", "bacc", "auc", "f1"]:
            vals = np.array([m[key] for m in all_metrics])
            print(f"{key}: {np.nanmean(vals):.4f} +/- {np.nanstd(vals):.4f}")


if __name__ == "__main__":
    main()
