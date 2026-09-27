"""
Structural smoke test for inference.py - builds tiny, fast, throwaway model
artifacts matching the real directory layout, then exercises the full
load_registry() -> score_text() pipeline against them.

This does NOT test model quality (the toy models are untrained/near-random).
It only proves inference.py's loading, adapters, and ensemble-averaging code
are wired correctly. Run it any time you edit inference.py, and again once
you've retrained real models under models/<category>/ to sanity-check the
real thing the same way (see the bottom of this file).

Usage:
    python test_inference.py
"""

import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import inference as inf


def build_category(root: Path, name: str, texts, labels_int, class_names):
    cat_dir = root / name
    cat_dir.mkdir(parents=True)
    (cat_dir / "labels.json").write_text(json.dumps(class_names))

    word_vec = TfidfVectorizer(max_features=200)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), max_features=200)
    X = hstack([word_vec.fit_transform(texts), char_vec.fit_transform(texts)])
    lr = LogisticRegression(max_iter=500).fit(X, labels_int)
    joblib.dump(
        {"model": lr, "word_vectorizer": word_vec, "char_vectorizer": char_vec},
        cat_dir / "logreg.joblib",
    )

    import keras as k
    from keras import Sequential
    from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Bidirectional, LSTM, Dense

    n_classes = len(class_names)
    seq_len = 16
    out_units = 1 if n_classes == 2 else n_classes
    out_act = "sigmoid" if n_classes == 2 else "softmax"
    loss = "binary_crossentropy" if n_classes == 2 else "sparse_categorical_crossentropy"

    for subdir, extra_layers in [
        ("textcnn", lambda: [Conv1D(8, 3, activation="relu"), GlobalMaxPooling1D()]),
        ("bilstm", lambda: [Bidirectional(LSTM(4))]),
    ]:
        d = cat_dir / subdir
        d.mkdir()
        vectorizer = inf.KerasTextVectorizer(max_tokens=200, output_sequence_length=seq_len)
        vectorizer.adapt(texts)
        X_seq = vectorizer(texts)

        model = Sequential(
            [k.Input(shape=(seq_len,), dtype="int32"), Embedding(200, 8)]
            + extra_layers()
            + [Dense(out_units, activation=out_act, dtype="float32")]
        )
        model.compile(optimizer="adam", loss=loss)
        model.fit(X_seq, np.asarray(labels_int, dtype="float32" if n_classes == 2 else "int32"),
                  epochs=1, verbose=0)

        model.save(str(d / f"{subdir}_single.keras"))
        joblib.dump(vectorizer, d / f"{subdir}_single_vectorizer.joblib")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    distil_dir = cat_dir / "distilbert" / "single"
    distil_dir.mkdir(parents=True)
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=n_classes)
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model.save_pretrained(distil_dir)
    tok.save_pretrained(distil_dir)


def main():
    root = Path(tempfile.mkdtemp())
    print("building toy artifacts in:", root)

    build_category(
        root, "toxic",
        ["you are great", "i hate you", "nice weather", "you are trash"],
        [0, 1, 0, 1],
        ["not_toxic", "toxic"],
    )
    build_category(
        root, "cyberbullying",
        ["ok comment", "religious slur here", "ageist remark here", "sexist remark here",
         "racist remark here", "other bullying text", "fine text", "another ok one"],
        [0, 1, 2, 3, 4, 5, 0, 0],
        ["not_cyberbullying", "religion", "age", "gender", "ethnicity", "other_cyberbullying"],
    )

    inf.CATEGORIES = ("toxic", "cyberbullying")  # only built 2 of 3 to keep this fast

    registry = inf.load_registry(root)
    scores, details = inf.score_text(registry, "i hate you, racist trash", return_details=True)

    print("\nscores:", scores)
    print("details:", json.dumps(details, indent=2))

    assert set(scores) == {"toxic", "cyberbullying"}
    assert all(0.0 <= v <= 1.0 for v in scores.values())
    assert len(details["cyberbullying"]["probabilities"]) == 6
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
