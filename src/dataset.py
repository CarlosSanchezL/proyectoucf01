
# src/dataset.py
import pickle
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset

class UCF101SkeletonDataset(Dataset):
    def __init__(self, pkl_path, split_name="train", selected_labels=None, seq_len=64):
        super().__init__()
        self.seq_len = seq_len
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.split = data["split"][split_name]
        self.annotations = data["annotations"]
        anno_by_id = {anno["frame_dir"]: anno for anno in self.annotations}

        samples = []
        for vid in self.split:
            anno = anno_by_id[vid]
            label = int(anno["label"])
            if selected_labels is not None and label not in selected_labels:
                continue
            samples.append(anno)

        if selected_labels is None:
            self.label_map = None
        else:
            self.label_map = {orig: i for i, orig in enumerate(sorted(selected_labels))}
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def _sample_or_pad(self, seq, seq_len):
        T, D = seq.shape
        if T == seq_len:
            return seq
        if T > seq_len:
            return seq[:seq_len]
        pad = np.zeros((seq_len - T, D), dtype=seq.dtype)
        return np.concatenate([seq, pad], axis=0)

    def __getitem__(self, idx):
        anno = self.samples[idx]
        keypoint = anno["keypoint"]
        H, W = anno["img_shape"]

        keypoint = keypoint[0]
        T, V, C = keypoint.shape
        keypoint_norm = keypoint.copy()
        keypoint_norm[..., 0] /= W
        keypoint_norm[..., 1] /= H

        seq = keypoint_norm.reshape(T, V*C).astype("float32")
        seq = self._sample_or_pad(seq, self.seq_len)

        label = int(anno["label"])
        if self.label_map is not None:
            label = self.label_map[label]

        return torch.from_numpy(seq), torch.tensor(label, dtype=torch.long)
