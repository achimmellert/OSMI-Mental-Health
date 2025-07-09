import os
import pandas as pd


filepath_base = "../data/edited/df_bis_auf_NLP_Spalten_rdy.pkl"
base_df = pd.read_pickle(filepath_base)

filepath_0 = "data_rdy_to_concat/Why_or_why_not_sentiment.pkl"
filepath_1 = "data_rdy_to_concat/Why_or_why_not_1_cluster_labels.pkl"
filepath_2 = "data_rdy_to_concat/Why_or_why_not_cluster_labels.pkl"
filepath_3 = "data_rdy_to_concat/Why_or_why_not1_sentiment.pkl"

files_to_concat = [filepath_0, filepath_1, filepath_2, filepath_3]



for filepath in files_to_concat:
    df_ = pd.read_pickle(filepath)
    base_df = pd.concat([base_df, df_], axis=1)


base_df = base_df.select_dtypes(exclude=["string", "object"])

for col in base_df.columns:
    try:
        base_df[col] = base_df[col].astype(float)
    except ValueError:
        print(f"Spalte '{col}' konnte nicht in float umgewandelt werden.")

base_df = base_df.dropna(axis=1, how="any")
ready_df = base_df.reset_index()

storage_path = "../data/edited/full_df_rdy_1.pkl"
pd.to_pickle(base_df, storage_path)
