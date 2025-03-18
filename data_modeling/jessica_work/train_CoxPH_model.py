import pickle
import numpy as np
import io
import sys
from load_table import get_train_test_split
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sksurv.metrics import concordance_index_censored
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def concordance_scorer(estimator,X,y):
    pred_risk = estimator.predict(X)
    event = y["event"]
    time = y["time"]
    c_index = concordance_index_censored(event, time, pred_risk)[0]
    return c_index

X_train, X_val, X_test, y_train, y_val, y_test = get_train_test_split()

features_to_remove = ["pde_id","bene_id","claim_start_year"]
X_train_og = X_train.copy()
X_test_og = X_test.copy()
X_val_og = X_val.copy()
X_train = X_train.drop(columns=features_to_remove,errors='ignore')
X_test= X_test.drop(columns=features_to_remove,errors="ignore")
X_val= X_val.drop(columns=features_to_remove,errors="ignore")

X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(X_train, y_train, test_size = 0.2, random_state=42)

param_grid = {"coxph__alpha": [0.1,0.5,1.0]}


#Training the model
coxph = CoxPHSurvivalAnalysis()
preprocessor = ColumnTransformer(transformers=[('num',Pipeline([('imputer',SimpleImputer(strategy='mean')),
                                                                ('scaler',StandardScaler())]), X_train.columns)])
pipeline = Pipeline([('preprocessor',preprocessor),('coxph',coxph)])
grid_search = GridSearchCV(pipeline, param_grid,cv=5,scoring=concordance_scorer,n_jobs=-1)
print("Starting grid search")
grid_search.fit(X_train_inner,y_train_inner)

#Best parameters
best_params = grid_search.best_params_
print("Best Hyperparamters:", best_params)
print(f"Best Concordance Index: {grid_search.best_score_:.3f}")

#Training with best parameters
best_pipeline = grid_search.best_estimator_
print("Training Optimized Model")
best_pipeline.fit(X_train, y_train)
print("Training completed")

#Evaluating on outer validation set
val_score_outer = best_pipeline.score(X_val,y_val)
print(f"Concordance index on outer validation: {val_score_outer:.3f}")

# Print size because model is getting too large to save locally
print(f"Serialized model size: ", len(pickle.dumps(best_pipeline, -1)) / (1024 * 1024), " MB")

# Save the model to a file so we don't need to train again
model_file_name = 'coxph_trained_model.pkl'
with open(model_file_name, 'wb') as file:
    pickle.dump(best_pipeline,file,-1)
print("Saved model")

#Evaluating
print("Scoring model on sample of test")
X_test_sampled = X_test[:min(10000,len(X_test))]
y_test_sampled = y_test[:min(10000,len(y_test))]
score = best_pipeline.score(X_test_sampled, y_test_sampled)
print(f"Concordance index: {score:.3f}")

#Feature importance

coxph_final = best_pipeline.named_steps['coxph']
X_test_transformed = best_pipeline.named_steps['preprocessor'].transform(X_test)
result = permutation_importance(coxph_final, X_test_transformed,y_test,n_repeats=10,random_state=42,n_jobs=-1)
feature_importance = result.importances_mean
features_names = X_train.columns

sorted_idx = np.argsort(feature_importance)[::-1]
print("Feature Importance:")
for i in sorted_idx:
    print(f"{features_names[i]}: {feature_importance[i]:.4f}")