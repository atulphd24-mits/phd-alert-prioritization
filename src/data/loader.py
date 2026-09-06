import pandas as pd
from pathlib import Path


class DatasetLoader:
    """Load CSV or Parquet datasets from the configured raw-data directory."""

    def __init__(self, config):
        self.data_root = Path(config.DATA_ROOT).expanduser()

    def load(self, dataset_path, sample_size=None):
        """Read supported files from a dataset directory.

        When ``sample_size`` is provided, return at most that many rows.
        """
        if sample_size is not None:
            if not isinstance(sample_size, int) or isinstance(sample_size, bool):
                raise TypeError("sample_size must be an integer or None")
            if sample_size <= 0:
                raise ValueError("sample_size must be greater than zero")

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
        rows_remaining = sample_size
        for file_path in files:
            if file_path.suffix.lower() == ".csv":
                read_options = {}
                if rows_remaining is not None:
                    read_options["nrows"] = rows_remaining
                frame = pd.read_csv(file_path, **read_options)
            else:
                frame = pd.read_parquet(file_path)
                if rows_remaining is not None:
                    frame = frame.head(rows_remaining)

            if not frame.empty:
                frames.append(frame)

            if rows_remaining is not None:
                rows_remaining -= len(frame)
                if rows_remaining <= 0:
                    break

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)