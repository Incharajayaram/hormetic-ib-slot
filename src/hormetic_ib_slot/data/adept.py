import json
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, List, Tuple


class ADEPTDataset(Dataset):
    """
    ADEPT object permanence benchmark loader (Kortmann et al., 2022).

    Expected directory structure:
        root_dir/
          sequences/
            seq_000001/
              frames/
                frame_000000.png
                frame_000001.png
                ...
              metadata.json
          splits/
            train.txt   (sequence IDs, one per line)
            val.txt
            test.txt
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        num_frames: int = 32,
        resolution: Tuple[int, int] = (64, 64),
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_frames = num_frames
        self.resolution = resolution  # (H, W)

        self.sequences = self._collect_sequences()
        if len(self.sequences) == 0:
            raise FileNotFoundError(
                f"No sequences found for split '{split}' under {self.root_dir}"
            )

    def _collect_sequences(self) -> List[Path]:
        split_file = self.root_dir / 'splits' / f'{self.split}.txt'
        seq_dir = self.root_dir / 'sequences'

        if split_file.exists():
            with open(split_file) as f:
                seq_ids = [line.strip() for line in f if line.strip()]
            sequences = [seq_dir / sid for sid in seq_ids if (seq_dir / sid).exists()]
        else:
            # Fall back to all subdirectories, sorted and split 80/10/10
            all_seqs = sorted(seq_dir.iterdir()) if seq_dir.exists() else []
            n = len(all_seqs)
            if self.split == 'train':
                sequences = all_seqs[:int(0.8 * n)]
            elif self.split == 'val':
                sequences = all_seqs[int(0.8 * n):int(0.9 * n)]
            else:
                sequences = all_seqs[int(0.9 * n):]
        return sequences

    def _load_metadata(self, seq_dir: Path) -> dict:
        meta_path = seq_dir / 'metadata.json'
        if not meta_path.exists():
            return {'sequence_id': seq_dir.name, 'num_frames': 0, 'objects': [], 'occluder_frames': []}
        with open(meta_path) as f:
            return json.load(f)

    def _load_frame(self, frame_path: Path) -> np.ndarray:
        img = cv2.imread(str(frame_path))
        if img is None:
            return np.zeros((self.resolution[0], self.resolution[1], 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.resolution[1], self.resolution[0]))
        return img

    def _find_occlusion_window(self, metadata: dict) -> Tuple[int, int]:
        """Return (occlusion_start, occlusion_end) frame indices."""
        occluder_frames = metadata.get('occluder_frames', [])
        objects = metadata.get('objects', [])

        if occluder_frames:
            start = min(occluder_frames)
            end = max(occluder_frames) + 1
            return start, end

        # Infer from object visibility
        for obj in objects:
            visible = obj.get('visible', [])
            if not visible:
                continue
            in_occlusion = False
            occ_start = None
            for i, v in enumerate(visible):
                if not v and not in_occlusion:
                    occ_start = i
                    in_occlusion = True
                elif v and in_occlusion:
                    return occ_start, i
        return 0, 0

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq_dir = self.sequences[idx]
        metadata = self._load_metadata(seq_dir)
        seq_id = metadata.get('sequence_id', seq_dir.name)

        frames_dir = seq_dir / 'frames'
        frame_paths = sorted(frames_dir.glob('frame_*.png')) if frames_dir.exists() else []

        total_frames = len(frame_paths)
        if total_frames == 0:
            # Return zeros
            video = torch.zeros(self.num_frames, 3, *self.resolution)
            occ_start, occ_end = 0, 0
            visibility = torch.zeros(self.num_frames, 1, dtype=torch.bool)
            return {
                'video': video,
                'sequence_id': seq_id,
                'object_visibility': visibility,
                'occlusion_start': occ_start,
                'occlusion_end': occ_end,
            }

        # Sample a contiguous clip of num_frames
        max_start = max(0, total_frames - self.num_frames)
        start = random.randint(0, max_start)
        selected_paths = frame_paths[start:start + self.num_frames]
        # Pad with last frame if needed
        while len(selected_paths) < self.num_frames:
            selected_paths.append(selected_paths[-1])

        frames = [self._load_frame(p) for p in selected_paths]
        frames_np = np.stack(frames, axis=0)  # (T, H, W, 3)
        video_tensor = torch.from_numpy(frames_np).float().permute(0, 3, 1, 2) / 255.0  # (T, 3, H, W)

        # Object visibility within the clip
        objects = metadata.get('objects', [])
        num_objects = len(objects)
        if num_objects > 0:
            vis_list = []
            for obj in objects:
                vis = obj.get('visible', [True] * total_frames)
                clip_vis = vis[start:start + self.num_frames]
                while len(clip_vis) < self.num_frames:
                    clip_vis.append(clip_vis[-1] if clip_vis else True)
                vis_list.append(clip_vis)
            visibility = torch.tensor(vis_list, dtype=torch.bool).T  # (T, num_objects)
        else:
            visibility = torch.ones(self.num_frames, 1, dtype=torch.bool)

        occ_start, occ_end = self._find_occlusion_window(metadata)
        # Adjust occlusion window to clip
        occ_start_clip = max(0, occ_start - start)
        occ_end_clip = min(self.num_frames, occ_end - start)

        return {
            'video': video_tensor,                      # (T, 3, H, W)
            'sequence_id': seq_id,
            'object_visibility': visibility,            # (T, num_objects) bool
            'occlusion_start': occ_start_clip,
            'occlusion_end': occ_end_clip,
        }


def adept_collate_fn(batch):
    videos = torch.stack([b['video'] for b in batch])
    seq_ids = [b['sequence_id'] for b in batch]
    occ_starts = [b['occlusion_start'] for b in batch]
    occ_ends = [b['occlusion_end'] for b in batch]
    # visibility tensors may differ in num_objects — keep as list
    visibilities = [b['object_visibility'] for b in batch]
    return {
        'video': videos,
        'sequence_id': seq_ids,
        'object_visibility': visibilities,
        'occlusion_start': occ_starts,
        'occlusion_end': occ_ends,
    }


def get_adept_loader(
    root_dir: str,
    split: str,
    batch_size: int,
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
    **dataset_kwargs,
) -> DataLoader:
    dataset = ADEPTDataset(root_dir=root_dir, split=split, **dataset_kwargs)
    if shuffle is None:
        shuffle = (split == 'train')
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=adept_collate_fn,
        pin_memory=True,
        drop_last=False,
    )
