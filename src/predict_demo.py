# src/predict_demo.py
import argparse
import torch

from dataset import UCF101SkeletonDataset
from models import SkeletonMLPBaseline, SkeletonLSTM


def load_model(model_type, input_dim, num_classes, checkpoint_path, device):
    if model_type == "mlp":
        model = SkeletonMLPBaseline(input_dim, num_classes)
    else:
        model = SkeletonLSTM(input_dim, num_classes)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_labels = [0, 1, 2, 3, 4]

    ds = UCF101SkeletonDataset(
        args.pkl_path,
        split_name=args.split,
        selected_labels=selected_labels,
        seq_len=args.seq_len,
    )

    input_dim = ds[0][0].shape[-1]
    num_classes = len(selected_labels)

    model = load_model(args.model_type, input_dim, num_classes, args.checkpoint, device)

    label_inv = {i: orig for i, orig in enumerate(sorted(selected_labels))}

    if args.video_name is not None:
        idx = None
        for i, anno in enumerate(ds.samples):
            if anno["frame_dir"] == args.video_name:
                idx = i
                break
        if idx is None:
            raise ValueError(f"Video '{args.video_name}' no existe en el split {args.split}")
        video = args.video_name
    else:
        idx = args.index
        video = ds.samples[idx]["frame_dir"]

    x, y = ds[idx]
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1).item()

    print(f"\n🔍 Predicción sobre video")
    print(f"Video: {video}")
    print(f"Label real (mapeada): {y.item()} → original: {label_inv[y.item()]}")
    print(f"Predicción: {pred} → original: {label_inv[pred]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=["mlp", "lstm"], default="lstm")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--video_name", type=str, default=None)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    main(args)
