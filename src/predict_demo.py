
# src/predict_demo.py
import argparse
import torch

from dataset import UCF101SkeletonDataset
from models import SkeletonMLPBaseline, SkeletonLSTM

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_labels = [0,1,2,3,4]

    ds = UCF101SkeletonDataset(args.pkl_path, split_name=args.split,
                               selected_labels=selected_labels, seq_len=args.seq_len)

    input_dim = ds[0][0].shape[-1]
    num_classes = len(selected_labels)

    model = SkeletonMLPBaseline(input_dim, num_classes) if args.model_type=="mlp" \
            else SkeletonLSTM(input_dim, num_classes)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    label_inv = {i: orig for i, orig in enumerate(sorted(selected_labels))}

    with torch.no_grad():
        for idx in [0,1,2,3,4]:
            x, y = ds[idx]
            x = x.unsqueeze(0).to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).item()
            print(f"Sample {idx} | True: {y.item()} → Orig {label_inv[y.item()]} | Pred: {pred} → Orig {label_inv[pred]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=["mlp","lstm"], default="lstm")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--seq_len", type=int, default=64)
    args = parser.parse_args()
    main(args)
