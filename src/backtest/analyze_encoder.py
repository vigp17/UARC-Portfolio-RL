import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_embeddings(path: str) -> pd.DataFrame:
    return pd.read_pickle(path)


def run_regime_classifier(df: pd.DataFrame) -> None:
    X = np.stack(df["embedding"].values)
    y = df["regime"].values

    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if len(np.unique(y_train)) < 2:
        print("Skipping classifier: training labels contain only one class.")
        return

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    print("Accuracy:", round(accuracy_score(y_test, preds), 4))
    print("\nClassification report:\n")
    print(classification_report(y_test, preds,zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))


def plot_pca(df: pd.DataFrame, save_path: str) -> None:
    X = np.stack(df["embedding"].values)
    y = df["regime"].values

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    for regime in np.unique(y):
        idx = y == regime
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], alpha=0.6, label=f"Regime {regime}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Encoder embeddings colored by regime")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()


if __name__ == "__main__":
    outputs = Path("outputs")
    files = sorted(outputs.glob("encoder_embeddings_*.pkl"))

    if not files:
        raise FileNotFoundError("No encoder embedding files found in outputs/")

    for f in files:
        print(f"\n=== {f.name} ===")
        df = load_embeddings(str(f))
        run_regime_classifier(df)
        plot_pca(df, str(outputs / f"{f.stem}_pca.png"))