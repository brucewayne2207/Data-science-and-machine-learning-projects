
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report
from sklearn.svm import SVC

df=pd.read_csv('https://raw.githubusercontent.com/amankharwal/Website-data/master/social.csv')
print(df.head())

x = np.array(df[["Age", "EstimatedSalary"]])
y = np.array(df[["Purchased"]])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
decisiontree = DecisionTreeClassifier()
logisticregression = LogisticRegression()
knearestclassifier = KNeighborsClassifier()
svm_classifier = SVC()
bernoulli_naiveBayes = BernoulliNB()
passiveAggressive = PassiveAggressiveClassifier()

knearestclassifier.fit(xtrain, ytrain)
decisiontree.fit(xtrain, ytrain)
logisticregression.fit(xtrain, ytrain)
passiveAggressive.fit(xtrain, ytrain)

data1 = {"Classification Algorithms": ["KNN Classifier", "Decision Tree Classifier",
                                       "Logistic Regression", "Passive Aggressive Classifier"],
      "Score": [knearestclassifier.score(x,y), decisiontree.score(x, y),
                logisticregression.score(x, y), passiveAggressive.score(x,y) ]}
score = pd.DataFrame(data1)
print(score)
