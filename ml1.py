from sklearn import tree
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
names = ['height','weight','age','male']
df=pd.read_csv("gender.csv")
array=df.values
X = array[:,0:4]
Y = array[:,3]

X_train, X_test, y_train, y_test = train_test_split(X, Y,  test_size=0.2, random_state=700)

clf=tree.DecisionTreeClassifier()
clf=clf.fit(X_train , y_train)
prediction=clf.predict(X_test)
print prediction

print "Score:", clf.score(X_test, y_test)