import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import GridSearchCV
import pickle

# Load and wrangle the data
def load_data():
    df = pd.read_csv("table.csv")
    df['srvc_dt'] = pd.to_datetime(df['srvc_dt'])
    df['next_fill_date'] = pd.to_datetime(df['next_fill_date'])
    df = df.dropna(subset=['next_fill_date']).copy()
    df['time_to_non_adherence'] = df['days_since_last_fill'].astype(float)
    df['event'] = df['non_adherence_event'].astype(bool)
    return df

# Exploratory Data Analysis (EDA)
def perform_eda(df):
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())
    
    # Calculate refill intervals
    df['refill_intervals'] = (df['next_fill_date'] - df['srvc_dt']).dt.days
    
    # Count prescriptions that were only prescribed once
    single_prescriptions = df['next_fill_date'].isna().sum()
    print(f"Number of prescriptions only prescribed once: {single_prescriptions}")
    
    # Dropping one-time prescriptions
    df = df[~df['next_fill_date'].isna()]
    
    # Distribution of refill intervals
    plt.figure(figsize=(10,5))
    sns.histplot(df['refill_intervals'].dropna(), bins=30, kde=True)
    plt.title('Distribution of Refill Intervals')
    plt.xlabel('Days Between Fills')
    plt.ylabel('Frequency')
    plt.show()
    
    # Correlation heatmap
    prescription_columns = ['next_fill_date', 'non_adherence_event', 'qty_dspnsd_num', 'zip_cd']
    prescription_data = df[prescription_columns]
    corr_matrix = prescription_data.corr()
    plt.figure(figsize=(12,6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidth=0.5, cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Correlation Heatmap for Prescription Behavior Variables')
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8)
    plt.tight_layout()
    plt.show()
    return df

def preprocess_data(df):
    df_agg = df.groupby("bene_id").agg({
        "time_to_non_adherence": "mean",
        "event": "max",
        "qty_dspnsd_num": "sum",
        "beneficiary_age_at_service": "mean",
        "zip_cd": "first",
        "drug_cvrg_status_cat": "first",
        "pde_id": "count",
        "days_since_last_fill": "std",
        "claim_start_year": "max"
    }).reset_index()
    
    df_agg["med_adherence_percent"] = (df_agg["qty_dspnsd_num"] / (df_agg["time_to_non_adherence"] + 1)) * 100
    df_agg["med_adherence_percent"] = df_agg["med_adherence_percent"].clip(0, 100)
    
    label_encoders = {}
    for col in ['drug_cvrg_status_cat', 'zip_cd']:
        le = LabelEncoder()
        df_agg[col] = le.fit_transform(df_agg[col])
        label_encoders[col] = le
    
    return df_agg

def split_data(df):
    y = Surv.from_arrays(event=df["event"], time=df["time_to_non_adherence"])
    X = df.drop(columns=["time_to_non_adherence", "event", "pde_id", "bene_id", "claim_start_year"])
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Model Training & Evaluation
def train_rsf(X_train, y_train, X_val, y_val):
    param_grid = {"n_estimators": [100, 150, 200], "min_samples_split": [5, 10], "max_depth": [5, 8, 10]}
    rsf = RandomSurvivalForest(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rsf, param_grid, cv=5, scoring=concordance_index_censored, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_rsf = grid_search.best_estimator_
    print(f"RSF Best Concordance Index: {grid_search.best_score_:.3f}")
    return best_rsf

def train_coxph(X_train, y_train, X_val, y_val):
    param_grid = {"coxph__alpha": [0.1, 0.5, 1.0]}
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), X_train.columns)
    ])
    pipeline = Pipeline([('preprocessor', preprocessor), ('coxph', CoxPHSurvivalAnalysis())])
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=concordance_index_censored, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_coxph = grid_search.best_estimator_
    print(f"CoxPH Best Concordance Index: {grid_search.best_score_:.3f}")
    return best_coxph

# Run process
df = load_data()
df = perform_eda(df)
df = preprocess_data(df)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

rsf_model = train_rsf(X_train, y_train, X_val, y_val)
coxph_model = train_coxph(X_train, y_train, X_val, y_val)

# Save models
with open("rsf_trained_model.pkl", "wb") as f:
    pickle.dump(rsf_model, f)
with open("coxph_trained_model.pkl", "wb") as f:
    pickle.dump(coxph_model, f)