#Naive Bayes prediction
#Faiz ul haque Zeya
#Data scientist and Associate professor
#CEO transys.
#Note:Run it in Anaconda.
#import data sets
from sklearn import datasets
iris = datasets.load_iris()

#Import Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

#Fit model
y = gnb.fit(iris.data, iris.target)

#Predict
yp=y.predict(iris.data)
#
print("Naive Bayes prediction\nFaiz ul haque Zeya\nData scientist and Associate professor\nCEO transys.")

print("Number of incorrect predictions Iris database")
incorrect=iris.target != yp
print(incorrect.sum()) 







































































