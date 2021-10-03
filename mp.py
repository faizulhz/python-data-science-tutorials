# Mobile price prediction using simple decision tree
# Prof. Faiz ul haque Zeya
# Freelance Data scientist
# Data set from kaggle 
# url: https://www.kaggle.com/iabhishekofficial/mobile-price-classification
import pandas as pd
from sklearn import tree

f=pd.read_csv("c:\\BU\\train.csv")
d=pd.read_csv("c:\\BU\\test.csv")
f.head()
x=f.iloc[:,1:20]
y=f.iloc[:,20:21]
print(x.head())
clf = tree.DecisioTreeClassifier()
clf = clf.fit(x, y)
xt=f.iloc[:,1:20]
yt=clf.predict(xt)
