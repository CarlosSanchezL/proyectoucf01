
# src/train.py
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import UCF101SkeletonDataset
from models import SkeletonMLPBaseline, SkeletonLSTM

def accuracy(preds, targets):
    return (preds == targets).float().mean().item()

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_acc = 0
    total_samples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        preds = logits.argmax(dim=1)
        batch = y.size(0)
        total_loss += loss.item() * batch
        total_acc += accuracy(preds, y) * batch
        total_samples += batch

    return total_loss / total_samples, total_acc / total_samples

def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_acc = 0
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

    selected_labels = [0,1,2,3,4]
    train_ds = UCF101SkeletonDataset(args.pkl_path, split_name=args.train_split,
                                     selected_labels=selected_labels, seq_len=args.seq_len)
    val_ds = UCF101SkeletonDataset(args.pkl_path, split_name=args.val_split,
                                   selected_labels=selected_labels, seq_len=args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    input_dim = train_ds[0][0].shape[-1]
    num_classes = len(selected_labels)

    model = SkeletonMLPBaseline(input_dim, num_classes) if args.model_type=="mlp" \
            else SkeletonLSTM(input_dim, num_classes)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best = 0
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch} | Train Acc {tr_acc:.4f} | Val Acc {val_acc:.4f}")

        if val_acc > best:
            best = val_acc
            out = Path(args.out_dir) / f"best_{args.model_type}.pt"
            out.parent.mkdir(exist_ok=True, parents=True)
            torch.save(model.state_dict(), out)
            print(f">>> Guardado: {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type=str, required=True)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--model_type", type=str, choices=["mlp","lstm"], default="lstm")
    parser.add_argument("--out_dir", type=str, default="../checkpoints")
    args = parser.parse_args()
    main(args)
