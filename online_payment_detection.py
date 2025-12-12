
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyexpat import features
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns',None)

df=pd.read_csv('credit_card_statement.csv')
print(df.head())
#
print(df.isnull().sum())

print(df.type.value_counts())

type=df.type.value_counts()
plt.figure(figsize=(10,7))
explode=[0,0.1]
plt.pie(df.type.value_counts(),labels=df.type.value_counts().index,autopct='%1.1f%%')
plt.title('Transaction Type')
plt.show()

le=LabelEncoder()
df['type']=le.fit_transform(df['type'])
df['nameOrig']=le.fit_transform(df['nameOrig'])
df['nameDest']=le.fit_transform(df['nameDest'])
print(df.head())
correlation=df.corr()
print(correlation['isFraud'].sort_values(ascending=False))

df['isFraud']=df['isFraud'].replace([0,1],['No_Fraud','Fraud'])
print(df.head())

x=np.array(df[['type','amount','oldbalanceOrg','newbalanceOrig']])
y=np.array(df[['isFraud']])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(model.score(x_test,y_test))
features_1=np.array([[4, 9000.60, 9000.60, 0.0]])
features_2=np.array([[1, 181.00, 181.0,0.00]])
print('Prediction_1',model.predict(features_1))
print('Prediction_2',model.predict(features_2))