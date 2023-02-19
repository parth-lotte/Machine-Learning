import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns
from sklearn.datasets import fetch_california_housing

from sklearn import linear_model
from sklearn.model_selection import train_test_split
housing = fetch_california_housing()




X=housing.data
y= housing.target

# print(X)
# print(X.shape)
print(y.shape)

# algorithm

l_reg= linear_model.LinearRegression()
plt.scatter(X.T[2],y)
plt.show()

X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

#train

model= l_reg.fit(X_train,y_train)
predictions= model.predict(X_test)
print("Predictions :", predictions )
print("R^2 value", l_reg.score(X,y))
print("coedd", l_reg.coef_)
print("Intercept",l_reg.intercept_ )