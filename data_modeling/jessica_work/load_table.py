import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sksurv.util import Surv
from sklearn.model_selection import train_test_split

def get_train_test_split():
  df = pd.read_csv("table.csv")

  df['srvc_dt'] = pd.to_datetime(df['srvc_dt'])
  df['next_fill_date'] = pd.to_datetime(df['next_fill_date'])

  df = df.dropna(subset=['next_fill_date']).copy()

  df.loc[:,'time_to_non_adherence'] = df['days_since_last_fill'].astype(float)
  df.loc[:,'event'] = df['non_adherence_event'].astype(bool)

  df_model = df.drop(columns = ['days_since_last_fill','non_adherence_event','bene_id','srvc_dt','next_fill_date'])

  categorical_columns = ['drug_cvrg_status_cat']
  label_encoders = {}

  for col in categorical_columns:
    le = LabelEncoder()
    df_model[col]=le.fit_transform(df_model[col])
    label_encoders[col]= le
  df_model.head()

  #Model building
  y = Surv.from_arrays(event=df_model["event"],time=df_model["time_to_non_adherence"])
  X = df_model.drop(columns =["time_to_non_adherence","event"])
#   print("columns in X:",X.dtypes)

  #Test/train split
  return train_test_split(X, y, test_size = 0.3, random_state = 42)