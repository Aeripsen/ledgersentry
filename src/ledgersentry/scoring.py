"""
Two interchangeable scoring paths behind one small interface.

PandasScorer is the straightforward path: build a 1-row DataFrame, push it
through the fitted ColumnTransformer, call the model. Profiling the serving
loop (see docs/adr/005-compiled-scoring.md) showed that on a single row this
path spends about two thirds of its time inside ColumnTransformer/pandas
indexing and only a fraction in the model itself.

CompiledScorer removes that overhead: at construction it reads the fitted
preprocessor ONCE (one-hot categories per categorical column, numeric column
order) and from then on builds the model's input matrix directly in numpy.
It changes no numbers - tests assert its transform output is exactly equal to
the ColumnTransformer's, decisions included - it only skips per-request pandas
machinery. The decision logic itself is not duplicated: both paths call the
same FraudDetector.decide.

The TransactionScorer protocol is the contract both implement. Its consumers
are real: the FastAPI service scores through it, the streaming replay scores
through it, and the benchmark harness (bench.py) drives BOTH implementations
through it to report the honest before/after.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .model import FraudDetector


@dataclass(frozen=True)
class ScoreResult:
    decision: str
    p_fraud: float
    confidence: float


@dataclass(frozen=True)
class BatchResult:
    decisions: np.ndarray  # dtype object, values in {"fraud", "legit", "review"}
    p_fraud: np.ndarray
    confidence: np.ndarray


class TransactionScorer(Protocol):
    """What the service, the stream replay, and the benchmark all need:
    score one transaction (a plain dict of features) or a whole frame."""

    @property
    def numeric_cols(self) -> list[str]: ...

    @property
    def categorical_cols(self) -> list[str]: ...

    def score_one(
        self, features: Mapping[str, Any], review_threshold: float = 0.0
    ) -> ScoreResult: ...

    def score_frame(
        self, df: pd.DataFrame, review_threshold: float = 0.0
    ) -> BatchResult: ...


def expected_columns(preprocessor: Any) -> tuple[list[str], list[str]]:
    """(numeric_cols, categorical_cols) the FITTED preprocessor expects, read
    off `transformers_` rather than hardcoded, so the scorer serves whichever
    dataset actually trained the artifact."""
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for name, _trans, cols in preprocessor.transformers_:
        if name == "num":
            numeric_cols = list(cols)
        elif name == "cat":
            categorical_cols = list(cols)
    return numeric_cols, categorical_cols


def _fitted_encoder(preprocessor: Any) -> Any | None:
    for name, trans, _cols in preprocessor.transformers_:
        if name == "cat":
            return trans
    return None


class PandasScorer:
    """The original serving path: 1-row DataFrame -> ColumnTransformer -> model.
    Kept as a working implementation (not a museum piece): it is the reference
    the compiled path is tested against, and the benchmark's honest baseline."""

    def __init__(self, preprocessor: Any, model: FraudDetector) -> None:
        self.preprocessor = preprocessor
        self.model = model
        self._numeric_cols, self._categorical_cols = expected_columns(preprocessor)

    @property
    def numeric_cols(self) -> list[str]:
        return self._numeric_cols

    @property
    def categorical_cols(self) -> list[str]:
        return self._categorical_cols

    def _frame_one(self, features: Mapping[str, Any]) -> pd.DataFrame:
        row: dict[str, Any] = {
            c: features.get(c, float("nan")) for c in self._numeric_cols
        }
        for c in self._categorical_cols:
            row[c] = features.get(c, "unknown")
        return pd.DataFrame([row])

    def score_one(
        self, features: Mapping[str, Any], review_threshold: float = 0.0
    ) -> ScoreResult:
        X = self.preprocessor.transform(self._frame_one(features))
        decision, p_fraud, confidence = self.model.decide(X, review_threshold)
        return ScoreResult(str(decision[0]), float(p_fraud[0]), float(confidence[0]))

    def score_frame(
        self, df: pd.DataFrame, review_threshold: float = 0.0
    ) -> BatchResult:
        X = self.preprocessor.transform(df)
        decision, p_fraud, confidence = self.model.decide(X, review_threshold)
        return BatchResult(decision, p_fraud, confidence)


class CompiledScorer:
    """Same contract, precompiled transform: one-hot maps and numeric column
    order are read from the fitted preprocessor once, then every request is a
    direct numpy fill. Missing numeric features become NaN (the model handles
    them natively - never a fake 0.0), unknown/missing categories one-hot to
    all zeros, exactly like OneHotEncoder(handle_unknown="ignore")."""

    def __init__(self, preprocessor: Any, model: FraudDetector) -> None:
        self.model = model
        self._numeric_cols, self._categorical_cols = expected_columns(preprocessor)

        # (column_name, {category_value: column_offset}) per categorical column,
        # in the exact output order the fitted ColumnTransformer produces:
        # the "cat" block first (if present), then the "num" passthrough block.
        self._onehot: list[tuple[str, dict[Any, int]]] = []
        offset = 0
        encoder = _fitted_encoder(preprocessor)
        if encoder is not None:
            for col, cats in zip(self._categorical_cols, encoder.categories_, strict=True):
                self._onehot.append(
                    (col, {cat: offset + j for j, cat in enumerate(cats)})
                )
                offset += len(cats)
        self._num_offset = offset
        self._width = offset + len(self._numeric_cols)

    @property
    def numeric_cols(self) -> list[str]:
        return self._numeric_cols

    @property
    def categorical_cols(self) -> list[str]:
        return self._categorical_cols

    def transform_one(self, features: Mapping[str, Any]) -> np.ndarray:
        """1 x width float64 matrix, exactly equal to what the fitted
        ColumnTransformer would produce for the same row (tested)."""
        vec = np.zeros((1, self._width), dtype=np.float64)
        for col, mapping in self._onehot:
            j = mapping.get(features.get(col, "unknown"))
            if j is not None:
                vec[0, j] = 1.0
        for k, col in enumerate(self._numeric_cols):
            value = features.get(col)
            vec[0, self._num_offset + k] = float("nan") if value is None else float(value)
        return vec

    def transform_frame(self, df: pd.DataFrame) -> np.ndarray:
        """Vectorized batch build: one comparison per known category, one
        column copy per numeric feature. No per-row pandas objects."""
        n = len(df)
        mat = np.zeros((n, self._width), dtype=np.float64)
        for col, mapping in self._onehot:
            if col in df.columns:
                values = df[col].to_numpy()
                for cat, j in mapping.items():
                    mat[values == cat, j] = 1.0
        for k, col in enumerate(self._numeric_cols):
            if col in df.columns:
                mat[:, self._num_offset + k] = df[col].to_numpy(dtype=np.float64)
            else:
                mat[:, self._num_offset + k] = np.nan
        return mat

    def score_one(
        self, features: Mapping[str, Any], review_threshold: float = 0.0
    ) -> ScoreResult:
        X = self.transform_one(features)
        decision, p_fraud, confidence = self.model.decide(X, review_threshold)
        return ScoreResult(str(decision[0]), float(p_fraud[0]), float(confidence[0]))

    def score_frame(
        self, df: pd.DataFrame, review_threshold: float = 0.0
    ) -> BatchResult:
        X = self.transform_frame(df)
        decision, p_fraud, confidence = self.model.decide(X, review_threshold)
        return BatchResult(decision, p_fraud, confidence)


def build_scorer(bundle: Mapping[str, Any]) -> CompiledScorer:
    """The serving-path scorer for a loaded artifact bundle."""
    return CompiledScorer(bundle["preprocessor"], bundle["model"])
