import pandas as pd
import numpy as np

a=pd.read_csv("e:\\a.csv",delimiter=',')
a=a.replace(np.nan, 0)
x=a.iloc[2:400,1:277]
y=a.iloc[2:400,279:280]
print(x.head())
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x, y)
xt=a.iloc[401:452,1:277]
yt=a.iloc[401:452,279:280]
y1=clf.predict(xt)
print(yt)
print(y1)