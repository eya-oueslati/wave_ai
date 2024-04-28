# -*- coding: utf-8 -*-
"""WAVEAI.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12oPRYhDY0OwrjoiTxJx1r9fPEXX2qM0-
"""
import pandas as pd

import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns

data = data = r'C:\Users\ADMIN\Downloads\ai4i2020 (2).csv'

Data = pd.read_csv(data)
Data.head()

Data.isna().sum()

Data.duplicated()

Data = Data.rename(columns={'Torque [Nm]': 'Torque'})
Data = Data.rename(columns={'Air temperature [K]': 'Air_temperature'})
Data = Data.rename(columns={'Process temperature [K]': 'Process_temperature'})
Data = Data.rename(columns={'Tool wear [min]': 'Tool_wear'})
Data = Data.rename(columns={'Machine failure': 'Machine_failure'})
Data = Data.rename(columns={'Rotational speed [rpm]': 'Rotational_speed'})
Data.head()

labelencoder=preprocessing.LabelEncoder()
Data["Type"]=labelencoder.fit_transform(Data["Type"])
Data

Dataf = Data.drop(['Product ID'] , axis=1)
Dataf

Dataf.info()

df = pd.DataFrame(Data)
df['Puissance']= (2 * 3.14159 * df['Torque'] * df['Rotational_speed']) / 60
df['deftemp']=abs(df['Process_temperature']-df['Air_temperature'])
df['surtension']=df['Tool_wear']*df['Torque']
df

seuil_min = 3500
seuil_max = 9000
sns.scatterplot(x="Puissance", y="PWF", data=df)
plt.axvline(seuil_min, color='red', linestyle='--', label='Seuil min')
plt.axvline(seuil_max, color='green', linestyle='--', label='Seuil max')

points_proches_min = df[(df['Puissance'] >= seuil_min - 100) & (df['Puissance'] <= seuil_min + 100)]
points_proches_max = df[(df['Puissance'] >= seuil_max - 100) & (df['Puissance'] <= seuil_max + 100)]
print( points_proches_min[['Puissance', 'PWF','UDI']].values)
print( points_proches_max[['Puissance', 'PWF','UDI']].values)

DL = df.loc[df['Type'] == 1, ['UDI', 'Type', 'surtension','OSF']]
DM = df.loc[df['Type'] == 2, ['UDI', 'Type', 'surtension','OSF']]
DH = df.loc[df['Type'] == 0, ['UDI', 'Type', 'surtension','OSF']]

seuil_max = 11000
sns.scatterplot(x="surtension", y="OSF", data=DL)
plt.axvline(seuil_max, color='green', linestyle='--', label='Seuil max')

seuil_max = 12000
sns.scatterplot(x="surtension", y="OSF", data=DM)
plt.axvline(seuil_max, color='red', linestyle='--', label='Seuil max')

seuil_max = 13000
sns.scatterplot(x="surtension", y="OSF", data=DH)
plt.axvline(seuil_max, color='green', linestyle='--', label='Seuil max')

DT= df.loc[df['Rotational_speed']<1380, ['UDI', 'deftemp','HDF']]
DT.head()

seuil_max = 8.6
sns.scatterplot(x="deftemp", y="HDF", data=DT)
plt.axvline(seuil_max, color='green', linestyle='--', label='Seuil max')

TW= df[['UDI','Tool_wear','TWF']]
seuil_min = 200
seuil_max = 240
sns.scatterplot(x="Tool_wear", y="TWF", data=TW)
plt.axvline(seuil_min, color='red', linestyle='--', label='Seuil min')
plt.axvline(seuil_max, color='green', linestyle='--', label='Seuil max')

points_proches_min = TW[(TW['Tool_wear'] <= seuil_min ) & (TW['TWF']==1)]
points_proches_max = TW[(TW['Tool_wear'] >=240) & (TW['TWF']==0)]
print(points_proches_min[['Tool_wear', 'TWF','UDI']].values)

print( points_proches_max[['Tool_wear', 'TWF','UDI']].values)

dt = df.loc[(df['TWF'] == 1) & (df['PWF'] == 1)]
dt

dc = df.loc[(df['TWF'] == 1) & (df['OSF'] == 1)]
dc

dk = df.loc[(df['OSF'] == 1) & (df['HDF'] == 1) & (df['PWF'] == 1)]
dk

dH = df.loc[(df['PWF'] == 1) & (df['OSF'] == 1)]
dH

dM = df.loc[(df['PWF'] == 1) & (df['HDF'] == 1)]
dM



# rules = [Rule(condition, classes[failure_mode]) for failure_mode, condition in conditions.items()]

dt = df.loc[(df['TWF'] == 1) ]
dt

def rule_based_classifier_with_conditions(features):
    conditions = {
    "TWF": lambda x: x["Tool_wear"] >= 200 and x["Tool_wear"] <= 240,
    "HDF": lambda x: abs(x["Process_temperature"] - x["Air_temperature"]) < 8.6 and x["Rotational_speed"] < 1380,
    "PWF": lambda x: (((x["Torque"] * x["Rotational_speed"]) * 2 * 3.141592653589793) / 60) < 3500 or (((x["Torque"] * x["Rotational_speed"])* 2 * 3.141592653589793 ) / 60) > 9000,
    "OSF": lambda x: (x["Type"] == 1 and (x["Tool_wear"] * x["Torque"]) > 11000) or (x["Type"] == 2 and (x["Tool_wear"] * x["Torque"]) > 12000) or (x["Type"] == 3 and (x["Tool_wear"] * x["Torque"]) > 13000),
    #"Machine_failure": lambda x: x["TWF"] == 1 or x["OSF"]==1 or x["PWF"]==1 or x["HDF"]==1
    }
    classes = {
    "TWF": "Tool wear failure",
    "HDF": "Heat dissipation failure",
    "PWF": "Power failure",
    "OSF": "Overstrain failure",
    #"Machine_failure": "Machine_failure"
    }
    detected_failures = []  # Liste pour stocker les pannes détectées

    # Parcourir chaque condition et vérifier si elle est satisfaite
    for failure_mode, condition in conditions.items():
        if condition(features):
            detected_failures.append(classes[failure_mode])

    # Retourner la liste des pannes détectées
    if detected_failures:
        return detected_failures
    else:
        return ['No failure detected']

# Example of using the classifier
sample_features = {
    "Tool_wear": 220,
    "Process_temperature": 200,
    "Air_temperature": 190,
    "Rotational_speed": 1200,
    "Torque": 30,
    "Type": 2
}
predicted_class = rule_based_classifier_with_conditions(sample_features)
print("Predicted class:", predicted_class)

import joblib
from joblib import dump
import pickle

# Open a file for writing
with open('model.pkl', 'wb') as f:
    # Serialize and dump the model function into the file
    pickle.dump(rule_based_classifier_with_conditions, f)


