import pandas as pd
from pathlib import Path


class DatasetLoader:
    """Load CSV/Parquet datasets from the configured raw-data directory."""

    def __init__(self, config):
        self.data_root = Path(config.DATA_ROOT).expanduser()

    def _resolve(self, dataset_path):
        dataset_path = Path(dataset_path).expanduser()
        if not dataset_path.is_absolute():
            dataset_path = self.data_root / dataset_path
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")
        return dataset_path

    def _files(self, dataset_path):
        files = sorted(
            path for path in dataset_path.rglob("*")
            if path.is_file() and path.suffix.lower() in {".csv", ".parquet"}
        )
        if not files:
            raise FileNotFoundError(f"No CSV or Parquet files found in {dataset_path}")
        return files

    def load(self, dataset_path, sample_size=None):
        """Read supported files; sample_size limits rows loaded into memory."""
        if sample_size is not None:
            if not isinstance(sample_size, int) or isinstance(sample_size, bool):
                raise TypeError("sample_size must be an integer or None")
            if sample_size <= 0:
                raise ValueError("sample_size must be greater than zero")

        dataset_path = self._resolve(dataset_path)
        files = self._files(dataset_path)

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

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def metadata(self, dataset_path):
        """
        Return dataset-level metadata without constructing a full DataFrame.

        CSV row counts are obtained by streaming lines; Parquet row counts use
        file metadata when pyarrow is available, with a pandas fallback.
        Results should be cached externally for very large collections.
        """
        dataset_path = self._resolve(dataset_path)
        files = self._files(dataset_path)

        details = []
        total_rows = 0

        for file_path in files:
            if file_path.suffix.lower() == ".csv":
                with file_path.open("rb") as f:
                    rows = sum(1 for _ in f)
                # One header row is assumed for ordinary CSV files.
                rows = max(rows - 1, 0)
            else:
                rows = None
                try:
                    import pyarrow.parquet as pq
                    rows = pq.ParquetFile(file_path).metadata.num_rows
                except Exception:
                    try:
                        rows = len(pd.read_parquet(file_path, columns=[]))
                    except Exception:
                        rows = None

            details.append({
                "file": str(file_path),
                "rows": rows,
                "size_bytes": file_path.stat().st_size,
                "format": file_path.suffix.lower().lstrip("."),
            })
            if rows is not None:
                total_rows += int(rows)

        unknown = any(item["rows"] is None for item in details)
        return {
            "dataset": dataset_path.name,
            "files": details,
            "total_rows": None if unknown else total_rows,
            "total_size_bytes": sum(item["size_bytes"] for item in details),
        }
