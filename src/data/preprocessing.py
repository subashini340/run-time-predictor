import pandas as pd
import numpy as np
from pathlib import Path

def load_raw(path: str = "data/raw/marathon_results.csv") -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False) 
    print(f"Loaded {len(df):,} rows and {df.shape[1]} columns.")
    return df

def time_to_minutes(t: str) -> float:
    if pd.isna(t):
        return np.nan
    parts = str(t).strip().split(":")
    if len(parts) == 3:
        try:
            h, m, s = map(int, parts)
            return h * 60 + m + s / 60
        except ValueError:
            return np.nan
    return np.nan

def parse_all_times(df: pd.DataFrame) -> pd.DataFrame:
    split_cols = ["5K", "10K", "15K", "20K", "25K",
                  "30K", "35K", "40K", "Half", "Official Time"]
    for col in split_cols:
        if col in df.columns:
            df[f"{col}_min"] = df[col].apply(time_to_minutes)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    if "Official Time_min" not in df.columns:
        print("ERROR: Could not find 'Official Time' column.")
        return df

    df = df.dropna(subset=["Official Time_min"])
    df = df[(df["Official Time_min"] > 120) & (df["Official Time_min"] < 480)]
    
    if "M/F" in df.columns:
        df["is_female"] = (df["M/F"].astype(str).str.upper().str.strip() == "F").astype(int)
    else:
        df["is_female"] = 0
    return df

def save_processed(df: pd.DataFrame, out_path: str = "data/processed/marathon_processed.parquet"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved processed data to {out_path} ({len(df):,} rows).")

def run():
    df = load_raw()
    df = parse_all_times(df)
    df = clean_data(df)
    save_processed(df)
    return df

if __name__ == "__main__":
    run()