import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#importing data

data= pd.read_csv('car.data')
print(data.head())

X=data[['buying',
        'maint',
        'safety']].values
Y=data[['class']]

print(X,Y)

# converting the data
Le= LabelEncoder()

for i in range (len(X[0])):

    X[:,i]=Le.fit_transform(X[:,i])
print(X)

#y

label_mapping={
    'unacc':0,
    'acc':1,
    'good':2,
    'Vgood':3
}

Y['class']=Y['class'].map(label_mapping)
Y=np.array(Y)
print(Y)

#Create Model
Knn= neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')

X_train,X_test,Y_train,Y_test= train_test_split(X,Y, test_size=0.2)

Knn.fit(X_train,Y_train)

prediction=Knn.predict(X_test)
acc=metrics.accuracy_score(Y_test,prediction)
print("prediction", prediction)
print("Accuracy", acc)

print("Actual value",Y[20])
print("Predicted Value", Knn.predict(X)[20])