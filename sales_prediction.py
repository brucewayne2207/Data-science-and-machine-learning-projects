
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

pd.set_option('display.max_columns',None)

df=pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/advertising.csv")
print(df.head())

print(df.isnull().sum())

plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(10,7))
sns.heatmap(df.corr(),annot=True)
plt.show()

x=df.drop(['Sales'],axis=1).to_numpy()
y=df['Sales'].to_numpy()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)
print('Mean squared error',mean_squared_error(y_test,y_pred))

df_new=pd.DataFrame(x_test)
df_new['Actual']=y_test
df_new['Predicted']=y_pred
df_new['Squared_error']=(y_test-y_pred)**2
print(df_new)