from __future__ import annotations

from pathlib import Path
import re
from typing import Optional

import pandas as pd

class NewsCorporaDataLoader:
	"""Load and query the parquet news dataset."""

	_STOPWORDS = {
		"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
		"in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will",
		"with", "this", "these", "those", "or", "not", "but", "if", "then", "than", "so",
		"we", "you", "they", "i", "me", "my", "our", "your", "their", "them", "his", "her",
	}

	_DATELINE_PATTERN = re.compile(
		r"^\s*[a-z][a-z\.\s-]*(?:,\s*[a-z][a-z\.\s-]*)?\s*(?:\([^)]+\))?\s*[\u2014\-]\s*",
		re.IGNORECASE,
	)

	_BOILERPLATE_PATTERNS = (
		re.compile(r"click\s+here\s+to\s+read\s+more\.?", re.IGNORECASE),
		re.compile(r"follow\s+us\s+on\s+twitter\.?", re.IGNORECASE),
		re.compile(r"reporting\s+by\s+[^.;]+;\s*editing\s+by\s+[^.;]+\.?", re.IGNORECASE),
	)

	def __init__(self, parquet_path: Optional[str] = None, remove_stopwords: bool = False) -> None:
		default_path = Path(__file__).resolve().parents[1] / "data" / "uci_news.snappy.parquet"
		self.parquet_path = Path(parquet_path) if parquet_path else default_path
		self.remove_stopwords = remove_stopwords
		self._dataset = self._load_dataset()

	def _load_dataset(self) -> pd.DataFrame:
		"""Load dataset once during object creation."""
		if not self.parquet_path.exists():
			raise FileNotFoundError(f"Dataset not found at: {self.parquet_path}")

		dataset = pd.read_parquet(self.parquet_path)
		dataset.columns = [col.lower() for col in dataset.columns]
		return self._preprocess_dataset(dataset)

	def _preprocess_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
		"""Apply normalization and news-specific cleanup to main_content only."""
		dataset = dataset.copy()
		if "main_content" in dataset.columns:
			dataset["main_content"] = dataset["main_content"].fillna("").map(self._preprocess_text)

		return dataset

	def _preprocess_text(self, text: str) -> str:
		"""Normalize and clean article text for downstream similarity tasks."""
		cleaned = str(text).lower()

		# Remove common dateline prefixes like: "city, state (publisher) -"
		cleaned = self._DATELINE_PATTERN.sub("", cleaned)

		for pattern in self._BOILERPLATE_PATTERNS:
			cleaned = pattern.sub(" ", cleaned)

		# Remove non-alphanumeric characters but keep internal hyphens in words.
		cleaned = re.sub(r"[^a-z0-9\s-]", " ", cleaned)
		cleaned = re.sub(r"(?<!\w)-|-(?!\w)", " ", cleaned)

		if self.remove_stopwords:
			tokens = [tok for tok in cleaned.split() if tok not in self._STOPWORDS]
			cleaned = " ".join(tokens)

		# Collapse tabs/newlines/multiple spaces.
		cleaned = re.sub(r"\s+", " ", cleaned).strip()
		return cleaned

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
