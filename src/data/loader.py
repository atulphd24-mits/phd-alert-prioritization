import pandas as pd
from pathlib import Path


class DatasetLoader:
    """Load CSV or Parquet datasets from the configured raw-data directory."""

    def __init__(self, config):
        self.data_root = Path(config.DATA_ROOT).expanduser()

    def load(self, dataset_path):
        """Read all supported files in a dataset directory into one DataFrame."""
        dataset_path = Path(dataset_path).expanduser()
        if not dataset_path.is_absolute():
            dataset_path = self.data_root / dataset_path

        if not dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")

        files = sorted(
            path
            for path in dataset_path.rglob("*")
            if path.is_file() and path.suffix.lower() in {".csv", ".parquet"}
        )
        if not files:
            raise FileNotFoundError(
                f"No CSV or Parquet files found in {dataset_path}"
            )

        frames = []
        for file_path in files:
            if file_path.suffix.lower() == ".csv":
                frames.append(pd.read_csv(file_path))
            else:
                frames.append(pd.read_parquet(file_path))

        return pd.concat(frames, ignore_index=True)