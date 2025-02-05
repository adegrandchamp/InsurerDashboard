<<<<<<< HEAD
'''
This code just loads the survival_table.csv file and assigns a test / training split.
I moved this into its own file so I could load the same test dataset without
having to retrain the model.
'''
=======
>>>>>>> cedc60708c4ab07855d054e92076d29f7ea2e6b5
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sksurv.util import Surv
from sklearn.model_selection import train_test_split

def get_train_test_split():
<<<<<<< HEAD
  df = pd.read_csv("survival_table.csv")
=======
  df = pd.read_csv("table.csv")
>>>>>>> cedc60708c4ab07855d054e92076d29f7ea2e6b5

  df['srvc_dt'] = pd.to_datetime(df['srvc_dt'])
  df['next_fill_date'] = pd.to_datetime(df['next_fill_date'])

  df = df.dropna(subset=['next_fill_date']).copy()

  df.loc[:,'time_to_non_adherence'] = df['days_since_last_fill'].astype(float)
  df.loc[:,'event'] = df['non_adherence_event'].astype(bool)

<<<<<<< HEAD
  #Aggregate by beneficiary 
  df_agg = df.groupby("bene_id").agg({
    "time_to_non_adherence": "mean",
    "event":"max",
    "qty_dspnsd_num":"sum",
    "beneficiary_age_at_service": "mean",
    "zip_cd": "first",
    "drug_cvrg_status_cat":"first",
    "pde_id":"count",
    "days_since_last_fill":"std",
    "claim_start_year":"max"
  }).reset_index()

  #Calculating MPR
  df_agg["med_adherence_percent"]=(df_agg["qty_dspnsd_num"]/(df_agg["time_to_non_adherence"]+1))*100
  df_agg["med_adherence_percent"] = df_agg["med_adherence_percent"].clip(0,100)
  
  df_model = df_agg


  categorical_columns = ['drug_cvrg_status_cat','zip_cd']
=======
  df_model = df.drop(columns = ['days_since_last_fill','non_adherence_event','bene_id','srvc_dt','next_fill_date'])

  categorical_columns = ['drug_cvrg_status_cat']
>>>>>>> cedc60708c4ab07855d054e92076d29f7ea2e6b5
  label_encoders = {}

  for col in categorical_columns:
    le = LabelEncoder()
    df_model[col]=le.fit_transform(df_model[col])
    label_encoders[col]= le
<<<<<<< HEAD
=======
  df_model.head()
>>>>>>> cedc60708c4ab07855d054e92076d29f7ea2e6b5

  #Model building
  y = Surv.from_arrays(event=df_model["event"],time=df_model["time_to_non_adherence"])
  X = df_model.drop(columns =["time_to_non_adherence","event"])
<<<<<<< HEAD

  # Test/train split
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)
  
  X_train, X_temp, y_train, y_temp = train_test_split(X,y,test_size=0.2,random_state=42)
  X_val, X_test, y_val, y_test, = train_test_split(X_temp, y_temp, test_size = 0.2, random_state= 42)

  return X_train, X_val, X_test, y_train, y_val, y_test
=======
#   print("columns in X:",X.dtypes)

  #Test/train split
  return train_test_split(X, y, test_size = 0.3, random_state = 42)
>>>>>>> cedc60708c4ab07855d054e92076d29f7ea2e6b5
