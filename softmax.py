#Softmax
#Faiz ul haque Zeya
#Data scientist and Associate professor
#CEO transys.
import numpy as np
scores = [3.0, 1.0, 0.2]

#first method
def softmax(x):
    sumexp=0
    for i in range(len(x)):
       sumexp = sumexp + np.exp(x[i])
    u=[]
    for i in range(len(x)):
        u.append(np.exp(x[i])/sumexp)
    return u

#second method
def softmax2(x):
        return np.exp(x)/np.sum(np.exp(x),axis=0)

print ("Scores are",scores)
print("Softmax2 scores are", softmax2(scores))
print("Softmax scores are" ,softmax(scores))
 