import numpy as np
import pandas as pd 
import os, sys 
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("parkinsons.data")
# print(df.head())

features = df.drop(['name', 'status'], axis=1).values
labels = df.loc[:,'status'].values

print(labels[labels==1].shape[0], labels[labels==0].shape[0])

scaler = MinMaxScaler((-1,1))
X=scaler.fit_transform(features)
y=labels

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=7)

model=XGBClassifier()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
print(accuracy_score(y_test,y_pred))



# Ornek Veri

sample_df = pd.DataFrame([{
    "MDVP:Fo(Hz)": 88.333,
    "MDVP:Fhi(Hz)": 112.24,
    "MDVP:Flo(Hz)": 84.072,
    "MDVP:Jitter(%)": 0.00505,
    "MDVP:Jitter(Abs)": 0.00006,
    "MDVP:RAP": 0.00254,
    "MDVP:PPQ": 0.0033,
    "Jitter:DDP": 0.00763,
    "MDVP:Shimmer": 0.02143,
    "MDVP:Shimmer(dB)": 0.197,
    "Shimmer:APQ3": 0.01079,
    "Shimmer:APQ5": 0.01342,
    "MDVP:APQ": 0.01892,
    "Shimmer:DDA": 0.03237,
    "NHR": 0.01166,
    "HNR": 21.118,
    "RPDE": 0.611137,
    "DFA": 0.776156,
    "spread1": -5.24977,
    "spread2": 0.391002,
    "D2": 2.407313,
    "PPE": 0.24974
}])


scaled_sample = scaler.transform(sample_df)

prediction = model.predict(scaled_sample)
prediction_prob = model.predict_proba(scaled_sample)

print(f"Tahmin Edilen Sınıf: {prediction[0]}")
print(f"Sağlıklı Olasılığı: {prediction_prob[0][0]:.4f}")
print(f"Parkinson Olasılığı: {prediction_prob[0][1]:.4f}") 