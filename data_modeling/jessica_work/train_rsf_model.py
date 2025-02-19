'''
This python file trains a Random Survival Forest model and saves the output
to a local file called rsf_trained_model.pkl. The model can be pretty large.
Once the .pkl file is created, you can run the model using run_rsf_model by
loading the .pkl file since training takes a long time.
'''
import pickle
import numpy as np
import io
import sys
from load_table import get_train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sklearn.impute import SimpleImputer
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = get_train_test_split()

#Training the model
rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=3, max_depth=10, random_state=42, n_jobs=-1, verbose=2)
print("Training")
rsf.fit(X_train, y_train)
print("Training completed")

# Print size because model is getting too large to save locally
print(f"Serialized model size: ", len(pickle.dumps(rsf, -1)) / (1024 * 1024), " MB")

# Save the model to a file so we don't need to train again
model_file_name = 'rsf_trained_model.pkl'
with open(model_file_name, 'wb') as file:
    pickle.dump(rsf, file,  -1)
print("Saved model")

#Evaluating
print("Scoring model on sample of test")
X_test_sampled = X_test[:1000]
y_test_sampled = y_test[:1000]
score = rsf.score(X_test_sampled, y_test_sampled)
print(f"Concordance index: {score:.4f}")