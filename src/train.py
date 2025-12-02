# src/train.py
import argparse
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import UCF101SkeletonDataset
from models import SkeletonMLPBaseline, SkeletonLSTM


def accuracy(preds, targets):
    return (preds == targets).float().mean().item()


def train_one_epoch(model, loader, optimizer, criterion, device, clip_grad=0.0):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        if clip_grad > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        preds = logits.argmax(dim=1)
        batch = y.size(0)
        total_loss += loss.item() * batch
        total_acc += accuracy(preds, y) * batch
        total_samples += batch

    return total_loss / total_samples, total_acc / total_samples


def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            preds = logits.argmax(dim=1)
            batch = y.size(0)
            total_loss += loss.item() * batch
            total_acc += accuracy(preds, y) * batch
            total_samples += batch

    return total_loss / total_samples, total_acc / total_samples


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    selected_labels = [0, 1, 2, 3, 4]

    train_ds = UCF101SkeletonDataset(args.pkl_path, split_name=args.train_split,
                                     selected_labels=selected_labels, seq_len=args.seq_len)
    val_ds = UCF101SkeletonDataset(args.pkl_path, split_name=args.val_split,
                                   selected_labels=selected_labels, seq_len=args.seq_len)
    test_ds = UCF101SkeletonDataset(args.pkl_path, split_name=args.test_split,
                                    selected_labels=selected_labels, seq_len=args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    input_dim = train_ds[0][0].shape[-1]
    num_classes = len(selected_labels)

    model = SkeletonMLPBaseline(input_dim, num_classes) if args.model_type == "mlp" \
            else SkeletonLSTM(input_dim, num_classes)

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"best_{args.model_type}.pt"

    print(f"\n🔧 Entrenando modelo: {args.model_type}")
    print(f"   lr={args.lr} | weight_decay={args.weight_decay} | clip_grad={args.clip_grad}\n")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, clip_grad=args.clip_grad
        )
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train loss: {tr_loss:.4f} acc: {tr_acc:.4f} | "
            f"Val loss: {val_loss:.4f} acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)
            print(f">>> Guardado mejor modelo: {ckpt_path} (val_acc={best_val_acc:.4f})")

    print("\n🔍 Evaluando mejor modelo...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    val_loss, val_acc = eval_model(model, val_loader, criterion, device)
    test_loss, test_acc = eval_model(model, test_loader, criterion, device)

    print("\n📊 Resultados finales:")
    print(f"  Mejor epoch: {best_epoch}")
    print(f"  Val Acc: {val_acc:.4f}")
    print(f"  Test Acc: {test_acc:.4f}\n")

    if args.save_results:
        results = {
            "model_type": args.model_type,
            "tag": args.tag,
            "best_epoch": best_epoch,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "weight_decay": args.weight_decay,
            "clip_grad": args.clip_grad,
        }

        results_path = out_dir / f"results_{args.model_type}_{args.tag}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"💾 Resultados guardados en: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pkl_path", type=str, required=True)
    parser.add_argument("--train_split", type=str, required=True)
    parser.add_argument("--val_split", type=str, required=True)
    parser.add_argument("--test_split", type=str, required=True)

    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--clip_grad", type=float, default=0.0)

    parser.add_argument("--model_type", type=str, choices=["mlp", "lstm"], default="lstm")

    parser.add_argument("--tag", type=str, default="run")  # 👈 NUEVO

    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--save_results", action="store_true")

    args = parser.parse_args()
    main(args)

