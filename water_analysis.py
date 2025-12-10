
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from pycaret.classification import *

pd.set_option('display.max_columns',None)

df=pd.read_csv('water_quality.csv')
print(df.head())

print('--------------------------------------')

df=df.dropna()
print(df.isnull().sum())

plt.figure(figsize=(15, 10))
sns.countplot(df.potability_score)
plt.title("Distribution of Unsafe and Safe Water")
plt.show()

figure=px.histogram(df,x="ph",
                    color='potability_score',
                    title="Factor affecting water quality:PH")
figure.show()

figure=px.histogram(df,x="Hardness",
                    color='potability_score',
                    title="Factor affecting water quality:Hardness")
figure.show()

figure=px.histogram(df,x="Solids",
                    color='potability_score',
                    title="Factor affecting water quality:Solids")
figure.show()

figure=px.histogram(df,x="Conductivity",
                    color='potability_score',
                    title="Factor affecting water quality:Conductivity")
figure.show()

figure=px.histogram(df,x="Turbidity",
                    color='potability_score',
                    title="Factor affecting water quality:Turbidity")
figure.show()

figure=px.histogram(df,x="Sulphide",
                    color='potability_score',
                    title="Factor affecting water quality:Sulphide")
figure.show()



figure=px.histogram(df,x="Ammonia",
                    color='potability_score',
                    title="Factor affecting water quality:Ammonia")
figure.show()
std=StandardScaler()
df['Potability']=std.fit_transform(df[['potability_score']])
print(df.head())
des=df.describe()
print(des)
correlation = df.corr()
correlation["ph"].sort_values(ascending=False)

clf = setup(df, target = ["Potability"], verbose = False, session_id = 786)
compare_models()

model = create_model("rf")
predict = predict_model(model, data=data)
predict.head()


