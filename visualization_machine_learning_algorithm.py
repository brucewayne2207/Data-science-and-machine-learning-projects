import pandas as pd
import numpy as np
from DS_5.sales_prediction import x_train
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as pg

data=pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/Advertising.csv")
print(data.head())

x=data["TV"]
y=data["Sales"]

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(xtrain,ytrain)
y_pred=model.predict(xtest)

fig=px.scatter(data,x,y_pred,color="Radio")
fig.show()