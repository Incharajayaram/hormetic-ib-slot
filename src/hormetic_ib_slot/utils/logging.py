import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class CSVLogger:
    """Append-mode CSV logger for training metrics."""

    def __init__(self, log_path: str, fieldnames: Optional[list] = None):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        self._initialized = self.log_path.exists() and self.log_path.stat().st_size > 0

    def log(self, row: dict):
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())

        write_header = not self._initialized
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction='ignore')
            if write_header:
                writer.writeheader()
                self._initialized = True
            writer.writerow(row)

    def read_all(self) -> list:
        if not self.log_path.exists():
            return []
        with open(self.log_path, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)


def setup_experiment_dir(base_dir: str, schedule_name: str, seed: int) -> Path:
    """
    Creates a timestamped experiment directory.
    Returns Path: base_dir/schedule_name/seed_<seed>_<YYYYMMDD_HHMMSS>/
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(base_dir) / schedule_name / f'seed_{seed}_{timestamp}'
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir
