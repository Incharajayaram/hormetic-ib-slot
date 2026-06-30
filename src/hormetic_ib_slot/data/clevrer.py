import os
import json
import glob
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, List, Dict, Any


class CLEVRERDataset(Dataset):
    """
    CLEVRER video dataset loader (Chen et al., 2020).

    Expected directory structure:
        root_dir/
          videos/
            train/video_00000-01000/video_00001.mp4 ...
            val/...
            test/...
          annotations/
            train/annotation_00001.json ...
            val/...
            test/...
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        num_frames: int = 16,
        frame_stride: int = 4,
        resolution: tuple = (64, 64),
        max_videos: Optional[int] = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.resolution = resolution  # (H, W)

        self.video_paths = self._collect_videos(max_videos)
        self.annotations = self._load_annotations()

        if len(self.video_paths) == 0:
            raise FileNotFoundError(
                f"No videos found for split '{split}' under {self.root_dir / 'videos'}"
            )

    def _collect_videos(self, max_videos: Optional[int]) -> List[Path]:
        video_dir = self.root_dir / 'videos' / self.split
        paths = sorted(video_dir.rglob('*.mp4'))
        if max_videos is not None:
            paths = paths[:max_videos]
        return paths

    def _load_annotations(self) -> Dict[int, Any]:
        ann_dir = self.root_dir / 'annotations' / self.split
        annotations = {}
        if not ann_dir.exists():
            return annotations
        for ann_file in ann_dir.glob('*.json'):
            try:
                with open(ann_file) as f:
                    data = json.load(f)
                vid_id = data.get('video_id', int(ann_file.stem.split('_')[-1]))
                annotations[vid_id] = data
            except (json.JSONDecodeError, ValueError):
                continue
        return annotations

    def _video_id_from_path(self, path: Path) -> int:
        stem = path.stem  # e.g. 'video_00001'
        try:
            return int(stem.split('_')[-1])
        except ValueError:
            return hash(str(path)) % 100000

    def _load_frames(self, video_path: Path, start_frame: int) -> Optional[np.ndarray]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        frame_indices = [start_frame + i * self.frame_stride for i in range(self.num_frames)]

        for idx in frame_indices:
            if idx >= total_video_frames:
                # Repeat last frame if video is shorter
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    cap.release()
                    return None
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    cap.release()
                    return None
                continue
            # BGR -> RGB, resize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.resolution[1], self.resolution[0]))
            frames.append(frame)

        cap.release()
        return np.stack(frames, axis=0)  # (T, H, W, 3)

    def _get_object_ids_for_frames(
        self, ann: Optional[dict], frame_indices: List[int]
    ) -> Optional[List[List[int]]]:
        if ann is None:
            return None
        frame_data = {f['frame_id']: f for f in ann.get('frames', [])}
        object_ids_per_frame = []
        for idx in frame_indices:
            frame_info = frame_data.get(idx, {})
            ids = [obj['id'] for obj in frame_info.get('objects', [])]
            object_ids_per_frame.append(ids)
        return object_ids_per_frame

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> dict:
        video_path = self.video_paths[idx]
        video_id = self._video_id_from_path(video_path)
        ann = self.annotations.get(video_id)

        # Determine valid start frame
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 128
        cap.release()

        required_span = (self.num_frames - 1) * self.frame_stride + 1
        max_start = max(0, total_frames - required_span)
        start_frame = random.randint(0, max_start) if max_start > 0 else 0

        frames_np = self._load_frames(video_path, start_frame)
        if frames_np is None:
            # Return zeros as fallback
            frames_np = np.zeros((self.num_frames, self.resolution[0], self.resolution[1], 3), dtype=np.uint8)

        # (T, H, W, 3) uint8 -> (T, 3, H, W) float32 [0, 1]
        video_tensor = torch.from_numpy(frames_np).float().permute(0, 3, 1, 2) / 255.0

        frame_indices = [start_frame + i * self.frame_stride for i in range(self.num_frames)]
        object_ids = self._get_object_ids_for_frames(ann, frame_indices)
        num_objects = len(ann['objects']) if ann is not None else 0

        return {
            'video': video_tensor,           # (T, 3, H, W)
            'video_id': video_id,
            'object_ids': object_ids,        # List[List[int]] or None
            'num_objects': num_objects,
        }


def clevrer_collate_fn(batch):
    videos = torch.stack([b['video'] for b in batch])
    video_ids = [b['video_id'] for b in batch]
    object_ids = [b['object_ids'] for b in batch]
    num_objects = [b['num_objects'] for b in batch]
    return {
        'video': videos,
        'video_id': video_ids,
        'object_ids': object_ids,
        'num_objects': num_objects,
    }


def get_clevrer_loader(
    root_dir: str,
    split: str,
    batch_size: int,
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
    **dataset_kwargs,
) -> DataLoader:
    dataset = CLEVRERDataset(root_dir=root_dir, split=split, **dataset_kwargs)
    if shuffle is None:
        shuffle = (split == 'train')
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=clevrer_collate_fn,
        pin_memory=True,
        drop_last=(split == 'train'),
    )
