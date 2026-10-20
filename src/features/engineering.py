import pandas as pd
import numpy as np

def compute_split_differentials(df: pd.DataFrame) -> pd.DataFrame:
    splits = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
    for i in range(1, len(splits)):
        prev_col = f"{splits[i-1]}_min"
        curr_col = f"{splits[i]}_min"
        if prev_col in df.columns and curr_col in df.columns:
            df[f"diff_{splits[i]}"] = df[curr_col] - df[prev_col]
    return df

def compute_pace_rate(df: pd.DataFrame) -> pd.DataFrame:
    split_cols = ["5K_min", "10K_min", "15K_min", "20K_min",
                  "25K_min", "30K_min", "35K_min", "40K_min"]
    available = [c for c in split_cols if c in df.columns]
    if len(available) < 2:
        df["pace_rate"] = np.nan
        return df

    x = np.arange(len(available))
    def slope(row):
        y = row[available].values.astype(float)
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return np.nan
        return np.polyfit(x[mask], y[mask], 1)[0]
        
    df["pace_rate"] = df.apply(slope, axis=1)
    return df

def run(df: pd.DataFrame) -> pd.DataFrame:
    df = compute_split_differentials(df)
    df = compute_pace_rate(df)
    return df

if __name__ == "__main__":
    print("Loading processed data...")
    df = pd.read_parquet("data/processed/marathon_processed.parquet")
    print("Engineering features...")
    df = run(df)
    output_path = "data/processed/marathon_featured.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Success! Saved to {output_path} ({len(df):,} rows).")