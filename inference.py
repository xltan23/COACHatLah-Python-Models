"""
Shared inference module for the toxic / inappropriate / cyberbullying
moderation models.

This is the single source of truth for "how a saved model turns text into a
probability" and "how the 4 base models per category get combined." Notebooks
and any future serving API should both import from here rather than
reimplementing prediction logic, so served scores can't silently drift from
what was validated during training.

Expected artifact layout (produced by toxic_models.ipynb / inappropriate_models.ipynb /
cyberbullying_models.ipynb):

    models/<category>/labels.json
    models/<category>/logreg.joblib
    models/<category>/textcnn/textcnn_single.keras
    models/<category>/textcnn/textcnn_single_vectorizer.joblib
    models/<category>/bilstm/bilstm_single.keras
    models/<category>/bilstm/bilstm_single_vectorizer.joblib
    models/<category>/distilbert/single/   (HF from_pretrained directory)

Usage:

    registry = load_registry("models")   # once, at process startup
    scores = score_text(registry, "some incoming text")
    # {"toxic": 0.9, "inappropriate": 0.6, "cyberbullying": 0.75}
"""

from __future__ import annotations

import os

# Must be set before `import keras` anywhere in the process - same
# requirement as the training notebooks.
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("USE_TF", "0")

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import joblib
import numpy as np
import torch
from scipy.sparse import hstack

import keras
from transformers import AutoModelForSequenceClassification, AutoTokenizer

assert keras.config.backend() == "torch", (
    f"Keras backend is {keras.config.backend()!r}, expected 'torch'."
)

CATEGORIES: Tuple[str, ...] = ("toxic", "inappropriate", "cyberbullying")
DISTILBERT_MAX_LENGTH = 128


# ---------------------------------------------------------------------------
# KerasTextVectorizer
#
# Must live in an importable module (not a notebook's __main__) so that
# joblib.load() can unpickle *_vectorizer.joblib files from any process that
# does `import inference` first - if the training notebooks still define this
# class inline, vectorizers they save will only unpickle inside a process
# whose __main__ happens to also define an identical class. Point the
# notebooks' training cells at `from inference import KerasTextVectorizer`
# before their next retrain to avoid that trap.
# ---------------------------------------------------------------------------
class KerasTextVectorizer:
    """No-TensorFlow stand-in for keras.layers.TextVectorization.

    Lowercase + strip punctuation, index 0 = padding, index 1 = OOV - matches
    Keras defaults, so Embedding(input_dim=max_tokens) lines up.
    """

    _PUNCT = re.compile(r"""[!"#$%&()*+,\-./:;<=>?@\[\\\]^_`{|}~']""")

    def __init__(self, max_tokens: int = 30000, output_sequence_length: int = 128):
        self.max_tokens = max_tokens
        self.output_sequence_length = output_sequence_length
        self.vocab: Dict[str, int] = {}

    @classmethod
    def _standardize(cls, text) -> List[str]:
        return cls._PUNCT.sub(" ", str(text).lower()).split()

    def adapt(self, texts) -> "KerasTextVectorizer":
        counter = Counter()
        for t in texts:
            counter.update(self._standardize(t))
        self.vocab = {
            word: i + 2
            for i, (word, _) in enumerate(counter.most_common(self.max_tokens - 2))
        }
        return self

    def __call__(self, texts) -> np.ndarray:
        texts = np.asarray(list(texts), dtype=object)
        out = np.zeros((len(texts), self.output_sequence_length), dtype="int32")
        for i, t in enumerate(texts):
            for j, w in enumerate(self._standardize(t)[: self.output_sequence_length]):
                out[i, j] = self.vocab.get(w, 1)
        return out

    transform = __call__

    @property
    def vocabulary_size(self) -> int:
        return self.max_tokens


# ---------------------------------------------------------------------------
# Model bundles
# ---------------------------------------------------------------------------
@dataclass
class LogisticBundle:
    model: object
    word_vectorizer: object
    char_vectorizer: object


@dataclass
class KerasBundle:
    model: object
    vectorizer: KerasTextVectorizer


@dataclass
class DistilBertBundle:
    model: object
    tokenizer: object
 

@dataclass
class CategoryModels:
    labels: List[str]
    lr: LogisticBundle
    textcnn: KerasBundle
    # bilstm: KerasBundle
    distilbert: DistilBertBundle


Registry = Dict[str, CategoryModels]


# ---------------------------------------------------------------------------
# Loading - call load_registry() exactly once per process and reuse the
# result. Every model here is expensive enough to load (DistilBERT weights,
# Keras models, TF-IDF vocabularies) that reloading per request would
# dominate request latency.
# ---------------------------------------------------------------------------
def _load_category(models_root: Path, category: str, device: torch.device) -> CategoryModels:
    cat_dir = models_root / category

    labels = json.loads((cat_dir / "labels.json").read_text())

    lr_saved = joblib.load(cat_dir / "logreg.joblib")
    lr = LogisticBundle(
        model=lr_saved["model"],
        word_vectorizer=lr_saved["word_vectorizer"],
        char_vectorizer=lr_saved["char_vectorizer"],
    )

    textcnn = KerasBundle(
        model=keras.models.load_model(str(cat_dir / "textcnn" / "textcnn_single.keras")),
        vectorizer=joblib.load(cat_dir / "textcnn" / "textcnn_single_vectorizer.joblib"),
    )

    # bilstm = KerasBundle(
    #     model=keras.models.load_model(str(cat_dir / "bilstm" / "bilstm_single.keras")),
    #     vectorizer=joblib.load(cat_dir / "bilstm" / "bilstm_single_vectorizer.joblib"),
    # )

    distil_dir = cat_dir / "distilbert" / "single"
    distil_model = AutoModelForSequenceClassification.from_pretrained(distil_dir)
    distil_model.to(device)
    distil_model.eval()
    distilbert = DistilBertBundle(
        model=distil_model,
        tokenizer=AutoTokenizer.from_pretrained(distil_dir),
    )

    # return CategoryModels(labels=labels, lr=lr, textcnn=textcnn, bilstm=bilstm, distilbert=distilbert)
    return CategoryModels(labels=labels, lr=lr, textcnn=textcnn, distilbert=distilbert)


def load_registry(models_root: Union[str, Path] = "models") -> Registry:
    """Load every category's 4 base models exactly once.

    Call this once at process startup (notebook kernel start, or API server
    startup) and pass the returned registry into score_text() - never
    reconstruct it per prediction.
    """
    models_root = Path(models_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {category: _load_category(models_root, category, device) for category in CATEGORIES}


# ---------------------------------------------------------------------------
# Per-model-type prediction adapters.
#
# Each returns a (n_texts, n_classes) probability array so the ensemble
# averaging code below doesn't need to know which model family it's calling.
# ---------------------------------------------------------------------------
def _predict_lr(bundle: LogisticBundle, texts: Sequence[str], n_classes: int) -> np.ndarray:
    word = bundle.word_vectorizer.transform(texts)
    char = bundle.char_vectorizer.transform(texts)
    raw = bundle.model.predict_proba(hstack([word, char]))

    if raw.shape[1] == n_classes:
        return raw

    # predict_proba's columns follow model.classes_, not a fixed 0..n-1
    # layout - realign explicitly in case a rare class was absent from
    # whatever split trained this LogisticRegression.
    full = np.zeros((raw.shape[0], n_classes), dtype=raw.dtype)
    full[:, bundle.model.classes_.astype(int)] = raw
    return full


def _predict_keras(bundle: KerasBundle, texts: Sequence[str]) -> np.ndarray:
    X = bundle.vectorizer(texts)
    raw = np.asarray(bundle.model.predict(X, verbose=0))
    if raw.shape[-1] == 1:
        # single sigmoid output trained directly against the positive-class
        # label -> expand to a 2-column [negative, positive] array so every
        # adapter returns the same (n_texts, n_classes) shape.
        positive = raw.reshape(-1)
        return np.stack([1.0 - positive, positive], axis=1)
    return raw  # already softmax over n_classes (cyberbullying)


def _predict_distilbert(bundle: DistilBertBundle, texts: Sequence[str]) -> np.ndarray:
    device = next(bundle.model.parameters()).device
    inputs = bundle.tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=DISTILBERT_MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = bundle.model(**inputs).logits
    return torch.softmax(logits, dim=-1).cpu().numpy()


# ---------------------------------------------------------------------------
# Per-category heterogeneous ensemble: LR + TextCNN + BiLSTM + DistilBERT,
# equal-weighted average.
# ---------------------------------------------------------------------------
def predict_category(category_models: CategoryModels, texts: Sequence[str]) -> np.ndarray:
    """Return the 4-model averaged (n_texts, n_classes) probability array."""
    n_classes = len(category_models.labels)
    probs = np.stack(
        [
            _predict_lr(category_models.lr, texts, n_classes),
            _predict_keras(category_models.textcnn, texts),
            # _predict_keras(category_models.bilstm, texts),
            _predict_distilbert(category_models.distilbert, texts),
        ],
        axis=0,
    )
    return probs.mean(axis=0)


def _category_score(labels: List[str], averaged: np.ndarray) -> float:
    """Reduce one category's averaged class-probability vector to a scalar.

    Binary categories (2 labels): P(positive class) = index 1.
    Multiclass (cyberbullying, index 0 = "not_*"): 1 - P(not_*).
    """
    if len(labels) == 2:
        return float(averaged[1])
    return float(1.0 - averaged[0])


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------
def score_text(
    registry: Registry,
    text: str,
    return_details: bool = False,
):
    """Score a single piece of text across all three moderation categories.

    Returns {"toxic": float, "inappropriate": float, "cyberbullying": float}.

    With return_details=True, also returns a second dict per category with
    its full label list and averaged per-class probabilities (e.g. to see
    *which* cyberbullying subtype was predicted) - useful for logging, not
    part of the basic contract.
    """
    scores: Dict[str, float] = {}
    details: Dict[str, dict] = {}

    for category in CATEGORIES:
        category_models = registry[category]
        averaged = predict_category(category_models, [text])[0]
        scores[category] = _category_score(category_models.labels, averaged)

        if return_details:
            details[category] = {
                "labels": category_models.labels,
                "probabilities": averaged.tolist(),
            }

    if return_details:
        return scores, details
    return scores


if __name__ == "__main__":
    import sys

    registry = load_registry("models")
    sample_text = sys.argv[1] if len(sys.argv) > 1 else "You are an idiot and I hate you"
    print(score_text(registry, sample_text))
