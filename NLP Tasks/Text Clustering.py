import logging
import pickle
import re
import time
from tqdm import tqdm
import optuna
from sklearn.metrics import silhouette_score
from umap import UMAP
import numpy as np
from pathlib import Path
import pandas as pd
from pandas import Series, DataFrame
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch
from transformers import pipeline
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import trustworthiness


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def load_column(filepath: Path, col: str) -> Series:

    logger.info("Datei einlesen...")

    try:
        df = pd.read_pickle(filepath)
        ser = df[col]
        ser = ser.fillna("Missing")

        logger.info(f"Spalte {col} erfolgreich geladen.")
        return ser

    except FileNotFoundError as fe:
        logger.error("Die Datei wurde nicht gefunden")
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError) as e:
        logger.error(f"Datei ist korrupt oder kein gültiges Pickle-Format: \n{e}")


def embedding(ser: Series):
    """
    Converts text into numeric vectors.
    """
    ser = ser.fillna("Missing")

    text_list = ser.to_list()

    logger.info("Calculates embeddings...")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(text_list, convert_to_numpy=True, show_progress_bar=True)

    df_embeddings = pd.DataFrame(embeddings, index=ser.index)

    return df_embeddings


def find_best_umap(df: DataFrame) -> UMAP:
    """
    Performs an Optuna Study to find the best hyperparamter for UMAP-model.
    """
    X = df.to_numpy()

    # Funktion, die es zu optimieren gilt
    def objective(trial):
        params = {
            "n_components": trial.suggest_int("n_components", 2, 15),
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 20),
            "min_dist": trial.suggest_float("min_dist", 0.0, 0.3),
        }

        umap = UMAP(metric="cosine", random_state=42, **params)

        reduced_X = umap.fit_transform(X)

        score = trustworthiness(X, reduced_X, metric="cosine")

        return score

    # Optuna Study erzeugen
    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.HyperbandPruner()
    )

    logger.info("Führt Optuna Study aus...")
    # Study ausführen
    study.optimize(objective, n_trials=100, timeout=600, n_jobs=-1)

    best_params = study.best_params
    final_model = UMAP(metric="cosine", random_state=42, **best_params)

    return final_model


def reduce_dimensionality(df: DataFrame, model: UMAP):
    """
    Reduces the dimensionality of the embeddings by using an umap-model.
    """

    umap = model

    logger.info("Reduces the dimensionality using the optimized umap model...")

    reduced_embeddings = umap.fit_transform(df)

    n_dims = reduced_embeddings.shape[1]
    new_cols = [f"dim_{i}" for i in range(n_dims)]

    reduced_embeddings_df = pd.DataFrame(
        reduced_embeddings, columns=new_cols, index=df.index
    )

    return reduced_embeddings_df


def find_best_kmeans(df: DataFrame) -> KMeans:
    """
    Finds the best Hyperparameter for KMeans (n_clusters).
    """
    X = df.to_numpy()  # (n_samples x n_features)
    k_range = range(2, 11)
    scores = []

    for k in tqdm(k_range, desc="Evaluating KMeans cluster sizes"):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X)

        score = silhouette_score(X, kmeans.labels_, metric="cosine")
        scores.append(score)

    optimal_k = k_range[np.argmax(scores)]

    logger.info("Best KMeans was found. Now trains with exact this KMeans model...")

    final_model = KMeans(n_clusters=optimal_k, random_state=42)

    return final_model


def kmeans_clustering(df: DataFrame, model: KMeans, cluster_name: str):
    """
    Finds the Clusters for the reduced embeddings.
    """
    X = df.to_numpy()
    kmeans = model.fit(X)

    cluster_labels = model.labels_

    logger.info("Traning beendet. Clusters als Series gespeichert! 🤗")

    return pd.Series(cluster_labels, name=cluster_name, index=df.index)


def main():

    start = time.time()

    filepath = Path.cwd().parent / "data" / "edited" / "df_alle_OHE_und_OE_Fragen.pkl"
    col = "Why or why not?"
    ser = load_column(filepath, col)

    embedded = embedding(ser)

    umap_model = find_best_umap(embedded)

    reduced = reduce_dimensionality(embedded, umap_model)

    kmeans_model = find_best_kmeans(reduced)

    cluster_name = "Why_or_why_not_Cluster"
    cluster_labels = kmeans_clustering(reduced, kmeans_model, cluster_name=cluster_name)

    cluster_labels.to_pickle("data_rdy_to_concat/Why_or_why_not_cluster_labels.pkl")

    end = time.time()
    print(f"\n\n=======================\n Beendet nach {end-start:.1f} Sekunden! 🤩")

if __name__ == "__main__":
    main()
