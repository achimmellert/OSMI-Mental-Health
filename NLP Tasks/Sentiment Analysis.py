# Sentiment Analysis

import logging
import pickle
import re
import time

import numpy as np
from pathlib import Path
import pandas as pd
from pandas import Series
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch
from transformers import pipeline
import spacy

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


def summarization(ser: Series) -> Series:

    logger.info("Summarization durchführen...")

    try:
        nlp = spacy.load("en_core_web_sm")
        summarizer = pipeline(
            task="summarization",
            model="facebook/bart-large-cnn",
            device=0,
            use_fast=True,  # using Rust insted of Python
        )

        def get_token_length(row):
            tokens = nlp.tokenizer(
                row
            )  # Normalerweise erstellt nlp ein Doc-Objekt mit mehr Informationen als nur Tokens
            return len(tokens)

        def get_summarized_text(text):
            summarized_text = summarizer(text, max_length=400, do_sample=False)[0][
                "summary_text"
            ]
            return summarized_text

        # .apply() führt die Iteration durch, lambda wendet ihre Funktion auf jedem Element an
        ser = ser.apply(
            lambda row: (
                row if get_token_length(row) <= 400 else get_summarized_text(row)
            )
        )
        return ser

    except Exception as e:
        logger.error(f"Error occured at summarization: {e}")
        raise


def sentiment_analysis(ser: Series) -> Series:
    """
    Führt Sentiment-Analyse durch.
    Gibt 0 (negativ), 1 (neutral), 2 (positiv) zurück.
    """

    logger.info("Sentiment Analysis durchführen...")

    try:
        sentiment_analyzer = pipeline(
            task="sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment",
            device=0,
        )

        # Liste von Texten
        texts = ser.tolist()

        results = sentiment_analyzer(texts, batch_size=32)

        # Label-zu-Int Mapping
        label_map = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}

        # Label extrahieren und kodieren
        encoded_results = list(label_map[result["label"]] for result in results)

        return Series(encoded_results, index=ser.index, name=ser.name)

    except Exception as e:
        logger.error(f"Error occured during sentiment analysis: {e}")
        raise


def main():

    start = time.time()

    filepath = Path.cwd().parent / "data" / "edited" / "df_alle_OHE_und_OE_Fragen.pkl"
    col_name = "Why or why not?"
    ser_default = load_column(filepath, col_name)

    ser = summarization(ser_default)
    ser_rdy = sentiment_analysis(ser)
    ser_rdy.name = "Why_or_why_not_sentiment"

    end = time.time()

    logger.info(
        f"Success: Spalte '{col_name}' erfolgreich verarbeitet nach {end - start:.1f} Sekunden.\n"
    )
    print(ser_rdy.head(10))

    filename = f"{re.sub(r'[^\w\s]', '', col_name).replace(' ', '_')}.pkl"
    pd.to_pickle(ser_rdy, filename)


if __name__ == "__main__":
    main()
