from configparser import ParsingError
from itertools import combinations
from pathlib import Path
import mlflow
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from pandas import DataFrame, Series
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from umap import UMAP
from yellowbrick.cluster import SilhouetteVisualizer


def load_data(path: Path) -> DataFrame:
    """
    Lädt den bereits encodeten DataFrame aus einer Pickle-Datei.
    """
    try:
        df = pd.read_pickle(path).reset_index(drop=True)

        return df

    except (FileNotFoundError, ParsingError, ValueError) as e:
        raise RuntimeError(f"Fehler beim Laden der Daten: {e}")


def find_best_params(data: DataFrame) -> dict:

    X = data.to_numpy()

    def objective(trial):
        """
        Funktion, die es zu maximieren gilt mit einer Optuna-Study.
        n_clusters beginn bei 4, damit das Ergebnis aussagekräftiger ist.
        """
        umap_params = {
            "n_components": trial.suggest_int("n_components", 2, 20),
            "n_neighbors": trial.suggest_int("n_neighbors", 2, 20),
            "metric": trial.suggest_categorical(
                "metric", ["euclidean", "cosine", "hamming", "jaccard"]
            ),
            "min_dist": trial.suggest_float("min_dist", 0.1, 1.0),
        }

        n_clusters = trial.suggest_int("n_clusters", 4, 15)

        umap = UMAP(random_state=42, **umap_params)

        X_reduced = umap.fit_transform(X)

        labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X_reduced)

        score = silhouette_score(X_reduced, labels, metric=umap_params["metric"])

        return score

    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=200)

    return study.best_params


def train_best_model(best_params: dict, df):
    """
    Trainiert ein UMAP + KMeans Modell mit den besten Parametern.

    Args:
        best_params (dict): Die besten UMAP- und KMeans-Parameter.
        df (DataFrame): Der Input-DataFrame.

    Returns:
        Tuple[np.ndarray, np.ndarray, KMeans]: Die reduzierten Daten, die Labels und das KMeans-Modell.
    """
    X = df.to_numpy()

    n_components = best_params["n_components"]
    n_neighbors = best_params["n_neighbors"]
    metric = best_params["metric"]
    min_dist = best_params["min_dist"]
    n_clusters = best_params["n_clusters"]

    umap = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        metric=metric,
        min_dist=min_dist,
        random_state=42,
    )
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    X_reduced = umap.fit_transform(X)
    labels = kmeans.fit_predict(X_reduced)

    return X_reduced, labels, kmeans


def cluster_plot(X, labels):
    pairs = list(combinations(range(9), 2))[:12]  # nur die ersten 12 Kombinationen
    n_plots = len(pairs)
    cols = 4
    rows = (n_plots // cols) + int(n_plots % cols > 0)

    fig = plt.figure(figsize=(4 * cols, 3 * rows))
    gs = plt.GridSpec(rows, cols)

    for i, (x_idx, y_idx) in enumerate(pairs):
        row = i // cols
        col = i % cols
        ax = fig.add_subplot(gs[row, col])
        ax.scatter(X[:, x_idx], X[:, y_idx], c=labels, cmap="viridis", s=10)
        ax.set_title(f"dim{x_idx + 1} vs dim{y_idx + 1}")
        ax.set_xlabel(f"dim{x_idx + 1}")
        ax.set_ylabel(f"dim{y_idx + 1}")

    plt.tight_layout()
    fig.suptitle("Cluster Plot")
    plt.savefig("../Plots/Zwölf Komponenten kombiniert.png", dpi=300)
    plt.show()


def silhoutte_plot(model, X):

    visualizer = SilhouetteVisualizer(model, colors="yellowbrick")
    visualizer.fit(X)
    visualizer.finalize()

    plt.tight_layout()
    plt.title("Silhoutte Plot")
    plt.savefig("../Plots/Silhoutte Plot.png", dpi=300)
    plt.show()


def main():

    filepath = Path.cwd().parent / "data" / "edited" / "full_df_rdy_1.pkl"
    df = load_data(filepath)

    # Starte MLflow Run
    with mlflow.start_run(run_name="UMAP + KMeans Clustering"):

        # 1. Optuna-Study
        best_params = find_best_params(df)

        # 2. Trainiere Modell mit besten Parametern aus der Optuna-Study
        X_reduced, labels, kmeans = train_best_model(best_params, df)

        # 3. Berechne Silhouette Score
        score = silhouette_score(X_reduced, labels, metric=best_params["metric"])
        mlflow.log_metric("silhouette_score", score)

        # 4. Logge alle besten Parameter
        mlflow.log_params(best_params)

        # 5. Speichere und logge Cluster-Plot
        cluster_plot(X_reduced, labels)
        mlflow.log_artifact("../Plots/Zwölf Komponenten kombiniert.png")

        # 6. Silhouette-Plot
        silhoutte_plot(kmeans, X_reduced)
        mlflow.log_artifact("../Plots/Silhoutte Plot.png")

        # Modell speichern
        mlflow.sklearn.load_model(kmeans, "model_kmeans")


if __name__ == "__main__":
    main()
