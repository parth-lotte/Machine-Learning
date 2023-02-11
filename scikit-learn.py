from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split 

iris= datasets.load_iris()
# Split in features and Labels
X= iris.data
Y=iris.target

# print(X,Y)
print(X.shape)
print(Y.shape)

#Horus of study vs good/bad grades
# 10 difeerent students 
# train with 8 students
#predict with the remaining 2
#levelof accuracy 

X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.2)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

