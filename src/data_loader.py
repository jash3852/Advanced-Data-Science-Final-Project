from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

class NewsCorporaDataLoader:
	"""Load and query the parquet news dataset."""

	def __init__(self, parquet_path: Optional[str] = None) -> None:
		default_path = Path(__file__).resolve().parents[1] / "data" / "uci_news.snappy.parquet"
		self.parquet_path = Path(parquet_path) if parquet_path else default_path
		self._dataset = self._load_dataset()

	def _load_dataset(self) -> pd.DataFrame:
		"""Load dataset once during object creation."""
		if not self.parquet_path.exists():
			raise FileNotFoundError(f"Dataset not found at: {self.parquet_path}")

		dataset = pd.read_parquet(self.parquet_path)
		dataset.columns = [col.lower() for col in dataset.columns]
		return dataset

	def get_dataset(self) -> pd.DataFrame:
		"""Return the entire loaded dataset."""
		return self._dataset.copy()

	def get_dataset_by_category(self, category: str) -> pd.DataFrame:
		"""Return all rows for a specific category value."""
		if "category" not in self._dataset.columns:
			raise KeyError("Column 'category' is not present in the dataset.")
		return self._dataset[self._dataset["category"] == category].copy()

	def get_dataset_by_hostname(self, hostname: str) -> pd.DataFrame:
		"""Return all rows for a specific hostname value."""
		if "hostname" not in self._dataset.columns:
			raise KeyError("Column 'hostname' is not present in the dataset.")
		return self._dataset[self._dataset["hostname"] == hostname].copy()
