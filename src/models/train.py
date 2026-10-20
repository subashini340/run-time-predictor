import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

def load_processed(path: str = "data/processed/marathon_featured.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)

def train_finish_time_models(df: pd.DataFrame):
    target = "Official Time_min"
    features = ["5K_min", "Age", "is_female", "pace_rate"]
    
    df = df.dropna(subset=[target])
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    pipe_rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    print("Training Random Forest Model...")
    pipe_rf.fit(X_train, y_train)
    preds_rf = pipe_rf.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds_rf)
    r2 = r2_score(y_test, preds_rf)
    
    print(f"\n--- Model Performance ---")
    print(f"Random Forest | MAE: {mae:.2f} mins | R²: {r2:.3f}")

    os.makedirs("reports", exist_ok=True)
    summary = pd.DataFrame([{"Model": "Random Forest", "MAE": mae, "R2": r2}])
    summary.to_csv("reports/model_summary.csv", index=False)

    # Saving the model for Streamlit
    joblib.dump(pipe_rf, 'marathon_model.pkl')
    print("\nSuccess! Model saved as marathon_model.pkl")

if __name__ == "__main__":
    print("Loading featured data...")
    df = load_processed()
    
    target = "Official Time_min"
    features = ["5K_min", "Age", "is_female", "pace_rate"]
    df = df.dropna(subset=[target])
    X = df[features]
    y = df[target]
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    import joblib
    
    final_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(n_estimators=20, max_depth=10, random_state=42, n_jobs=-1))
    ])
    
    print("Training lightweight model...")
    final_model.fit(X, y)
    
    joblib.dump(final_model, 'marathon_model.pkl', compress=3)
    print("Success! Lightweight model saved as marathon_model.pkl!")