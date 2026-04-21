from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numba import njit, prange


# Large prime used by the MinHash permutation family.
_MERSENNE_PRIME = np.uint64((1 << 61) - 1)
_MAX_HASH = np.uint64((1 << 61) - 1)
_WORD_RE = re.compile(r"\w+")


@njit(cache=True)
def _compute_signature_numba(
    shingles: np.ndarray,
    coeff_a: np.ndarray,
    coeff_b: np.ndarray,
    prime: np.uint64,
) -> np.ndarray:
    """Compute one MinHash signature from a 1D array of shingle hashes."""
    num_hashes = coeff_a.shape[0]
    signature = np.empty(num_hashes, dtype=np.uint64)

    for i in range(num_hashes):
        signature[i] = _MAX_HASH

    if shingles.shape[0] == 0:
        return signature

    for i in range(num_hashes):
        a = coeff_a[i]
        b = coeff_b[i]
        current_min = _MAX_HASH

        for j in range(shingles.shape[0]):
            value = (a * shingles[j] + b) % prime
            if value < current_min:
                current_min = value

        signature[i] = current_min

    return signature


@njit(parallel=True, cache=True)
def _pairwise_signature_similarity(signatures: np.ndarray) -> np.ndarray:
    """Estimate pairwise Jaccard similarity from MinHash signatures."""
    n_docs = signatures.shape[0]
    n_hashes = signatures.shape[1]
    output = np.zeros((n_docs, n_docs), dtype=np.float64)

    for i in prange(n_docs):
        output[i, i] = 1.0
        for j in range(i + 1, n_docs):
            matches = 0
            for k in range(n_hashes):
                if signatures[i, k] == signatures[j, k]:
                    matches += 1
            sim = matches / n_hashes
            output[i, j] = sim
            output[j, i] = sim

    return output


@njit(cache=True)
def _signature_similarity(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    matches = 0
    n_hashes = sig_a.shape[0]
    for i in range(n_hashes):
        if sig_a[i] == sig_b[i]:
            matches += 1
    return matches / n_hashes


@dataclass
class PairSimilarityResult:
    left_index: int
    right_index: int
    estimated_jaccard: float
    exact_jaccard: Optional[float] = None


class NumbaMinHash:
    """
    MinHash implementation that uses Numba for the hot loop.

    Workflow:
    1. Normalize article text.
    2. Build k-shingles.
    3. Hash shingles into uint64 values.
    4. Use Numba to compute a MinHash signature for each document.
    """

    def __init__(self, num_hashes: int = 128, shingle_size: int = 5, seed: int = 42) -> None:
        if num_hashes <= 0:
            raise ValueError("num_hashes must be positive.")
        if shingle_size <= 0:
            raise ValueError("shingle_size must be positive.")

        self.num_hashes = num_hashes
        self.shingle_size = shingle_size
        self.seed = seed
        self._coeff_a, self._coeff_b = self._build_hash_family(num_hashes, seed)

    @staticmethod
    def _build_hash_family(num_hashes: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        coeff_a = rng.integers(1, int(_MERSENNE_PRIME - 1), size=num_hashes, dtype=np.uint64)
        coeff_b = rng.integers(0, int(_MERSENNE_PRIME - 1), size=num_hashes, dtype=np.uint64)
        return coeff_a, coeff_b

    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.lower()
        tokens = _WORD_RE.findall(text)
        return " ".join(tokens)

    @staticmethod
    def _stable_uint64_hash(text: str) -> np.uint64:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="little", signed=False)
        # Keep values below the prime used in the permutation family.
        return np.uint64(value % int(_MERSENNE_PRIME))

    def text_to_shingle_hashes(self, text: str) -> np.ndarray:
        normalized = self.normalize_text(text)
        words = normalized.split()

        if len(words) < self.shingle_size:
            return np.empty(0, dtype=np.uint64)

        shingles = set()
        for i in range(len(words) - self.shingle_size + 1):
            shingle = " ".join(words[i : i + self.shingle_size])
            shingles.add(self._stable_uint64_hash(shingle))

        if not shingles:
            return np.empty(0, dtype=np.uint64)

        return np.array(sorted(shingles), dtype=np.uint64)

    def signature_from_text(self, text: str) -> np.ndarray:
        shingles = self.text_to_shingle_hashes(text)
        return _compute_signature_numba(shingles, self._coeff_a, self._coeff_b, _MERSENNE_PRIME)

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        signatures = np.empty((len(texts), self.num_hashes), dtype=np.uint64)
        for i, text in enumerate(texts):
            signatures[i] = self.signature_from_text(text)
        return signatures

    def estimate_similarity_matrix(self, texts: Sequence[str]) -> np.ndarray:
        signatures = self.fit_transform(texts)
        return _pairwise_signature_similarity(signatures)

    def compare_two_texts(self, left_text: str, right_text: str) -> PairSimilarityResult:
        left_signature = self.signature_from_text(left_text)
        right_signature = self.signature_from_text(right_text)
        estimated = _signature_similarity(left_signature, right_signature)
        exact = self.exact_jaccard(left_text, right_text)
        return PairSimilarityResult(
            left_index=0,
            right_index=1,
            estimated_jaccard=float(estimated),
            exact_jaccard=float(exact),
        )

    def top_similar_pairs(
        self,
        texts: Sequence[str],
        top_k: int = 10,
        exact_check: bool = False,
    ) -> List[PairSimilarityResult]:
        if len(texts) < 2:
            return []

        signatures = self.fit_transform(texts)
        sim_matrix = _pairwise_signature_similarity(signatures)

        candidates: List[PairSimilarityResult] = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                exact_score = None
                if exact_check:
                    exact_score = self.exact_jaccard(texts[i], texts[j])
                candidates.append(
                    PairSimilarityResult(
                        left_index=i,
                        right_index=j,
                        estimated_jaccard=float(sim_matrix[i, j]),
                        exact_jaccard=None if exact_score is None else float(exact_score),
                    )
                )

        candidates.sort(key=lambda item: item.estimated_jaccard, reverse=True)
        return candidates[:top_k]

    def exact_jaccard(self, left_text: str, right_text: str) -> float:
        left = set(self.text_to_shingle_hashes(left_text).tolist())
        right = set(self.text_to_shingle_hashes(right_text).tolist())

        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0

        intersection = len(left & right)
        union = len(left | right)
        return intersection / union

    def transform_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "main_content",
        id_column: Optional[str] = "id",
    ) -> pd.DataFrame:
        if text_column not in df.columns:
            raise KeyError(f"Column '{text_column}' not found in dataframe.")

        working_df = df.copy()
        working_df[text_column] = working_df[text_column].fillna("").astype(str)
        signatures = self.fit_transform(working_df[text_column].tolist())

        signature_cols = {
            f"mh_{i}": signatures[:, i]
            for i in range(self.num_hashes)
        }
        result = pd.DataFrame(signature_cols)

        if id_column and id_column in working_df.columns:
            result.insert(0, id_column, working_df[id_column].values)

        return result

    def find_candidate_pairs_in_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "main_content",
        id_column: str = "id",
        top_k: int = 10,
        exact_check: bool = True,
    ) -> pd.DataFrame:
        if text_column not in df.columns:
            raise KeyError(f"Column '{text_column}' not found in dataframe.")

        texts = df[text_column].fillna("").astype(str).tolist()
        pairs = self.top_similar_pairs(texts=texts, top_k=top_k, exact_check=exact_check)

        rows = []
        for item in pairs:
            left_id = df.iloc[item.left_index][id_column] if id_column in df.columns else item.left_index
            right_id = df.iloc[item.right_index][id_column] if id_column in df.columns else item.right_index
            rows.append(
                {
                    "left_index": item.left_index,
                    "right_index": item.right_index,
                    "left_id": left_id,
                    "right_id": right_id,
                    "estimated_jaccard": item.estimated_jaccard,
                    "exact_jaccard": item.exact_jaccard,
                }
            )

        return pd.DataFrame(rows)


if __name__ == "__main__":
    # Small self-test/demo.
    docs = [
        "Apple unveils new iPhone with updated camera and battery life.",
        "Apple reveals a new iPhone featuring a better camera and longer battery life.",
        "The Denver Nuggets won their basketball game last night.",
    ]

    minhash = NumbaMinHash(num_hashes=128, shingle_size=3, seed=42)
    print(minhash.find_candidate_pairs_in_dataframe(
        pd.DataFrame({"id": [1, 2, 3], "main_content": docs}),
        top_k=3,
        exact_check=True,
    ))
