import pickle
from load_table import get_train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

rsf = None
model_file_name = 'coxph_trained_model.pkl'
with open(model_file_name, 'rb') as file:  
    best_pipeline = pickle.load(file)

print("Model parameters:",best_pipeline.named_steps['coxph'].get_params())
X_train, X_val, X_test, y_train, y_val, y_test= get_train_test_split()

features_to_remove = ["pde_id","bene_id","claim_start_year"]
X_train_og = X_train.copy()
X_test_og = X_test.copy()
X_val_og = X_val.copy()
X_train = X_train.drop(columns=features_to_remove,errors='ignore')
X_test= X_test.drop(columns=features_to_remove,errors="ignore")
X_val= X_val.drop(columns=features_to_remove,errors="ignore")


print("Scoring model on validation set")
val_score = best_pipeline.score(X_val,y_val)
print(f"Concordance index on validation: {val_score:.3f}")

print("Scoring model on test set")
test_score = best_pipeline.score(X_test, y_test)
print(f"Concordance index: {test_score:.3f}")

df = pd.concat([X_test_og.reset_index(drop=True),pd.DataFrame(X_test,index=X_test.index).reset_index(drop=True)],axis=1)

scaler = MinMaxScaler()
df["risk_score"] = best_pipeline.predict(X_test)
scaler.fit(df[["risk_score"]])

df["normalized_risk_score"] = scaler.transform(df[["risk_score"]])
df["time_to_non_adherence"] = scaler.transform(df[["risk_score"]])
#print(df["normalized_risk_score"])

df["time_non_adherence"] = y_test["time"]
df["event"] = y_test["event"]

df.to_csv('coxph_model_results.csv',index=False)
print("Data saved")
#print(df.head())
