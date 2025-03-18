##EDA for Time to Medication Non-adherence Event

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('table.csv')

#Basic Info
print(df.info())
print(df.describe())

#Missing info
print(df.isnull().sum())

#Datetime format
df['srvc_dt'] = pd.to_datetime(df['srvc_dt'])
df['next_fill_date']=pd.to_datetime(df['next_fill_date'])

#Refill intervals
df['refill_intervals']=(df['next_fill_date'] - df['srvc_dt']).dt.days

#Count Prescriptions that were Only Prescribed Once
single_prescriptions = df['next_fill_date'].isna().sum()
print(f"Number of prescriptions only prescribed once: {single_prescriptions}")

#Dropping One-Time Prescriptions
df = df[~df['next_fill_date'].isna()]

#Distribution of Refill Intervals
plt.figure(figsize=(10,5))
sns.histplot(df['refill_intervals'].dropna(),bins=30,kde=True)
plt.title('Distribution of Refill Intervals')
plt.xlabel('Days Between Fills')
plt.ylabel('Frequency')
plt.show()

#Correlation
prescription_columns = ['next_fill_date','non_adherence_event','qty_dspnsd_num',
                        'zip_cd']
prescription_data=df[prescription_columns]
corr_matrix = prescription_data.corr()
plt.figure(figsize=(12,6))
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',fmt=".2f",
            linewidth=0.5,cbar_kws={'label':'Correlation Coefficient'})
plt.title('Correlation Heatmap for Prescription Behavior Variables')
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)
plt.tight_layout()
plt.show()
