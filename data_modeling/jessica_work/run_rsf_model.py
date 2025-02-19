import pickle
from load_table import get_train_test_split

rsf = None
model_file_name = 'rsf_trained_model.pkl'
with open(model_file_name, 'rb') as file:  
    rsf = pickle.load(file)

X_train, X_test, y_train, y_test = get_train_test_split()

print("Scoring model")
score = rsf.score(X_test, y_test)
print(f"Concordance index: {score:.4f}")