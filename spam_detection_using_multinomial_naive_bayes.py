import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

data=pd.read_csv("https://raw.githubusercontent.com/amankharwal/SMS-Spam-Detection/master/spam.csv", encoding= 'latin-1')
print(data.head(100))

data=data[['message','class']]
x=np.array(data['class'])
y=np.array(data['message'])
cv=CountVectorizer()
X=cv.fit_transform(x)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

mb=MultinomialNB()
mb.fit(X_train,y_train)

sample=input('Enter a message:')
data=cv.transform([sample]).toarray()
print(mb.predict(data))