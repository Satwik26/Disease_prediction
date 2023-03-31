import SVM
import Logistic_Regression as LR
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


diabetes_data = pd.read_csv("diabetes.csv")
feature = diabetes_data.drop('Outcome',axis=1)
target = diabetes_data['Outcome']
scaler = StandardScaler()
scaler.fit(feature)
standardized_data = scaler.transform(feature)
feature = standardized_data
x_train,x_test,y_train,y_test = train_test_split(feature,target,test_size=0.2,stratify=target,random_state=2)
classifier = SVM.Support_Vector(learning_rate=0.001,no_of_iteration=1000,lambda_param=0.01)
classifier.fit(x_train,y_train)
x_train_pred = classifier.predict(x_train)
training_data_acc = accuracy_score(y_train,x_train_pred)
#print(training_data_acc)
x_test_pred = classifier.predict(x_test)
test_data_acc = accuracy_score(y_test,x_test_pred)
#print(test_data_acc)


heart_data = pd.read_csv('heart_disease_data.csv')
features = heart_data.drop(columns='target',axis=1)
tar = heart_data['target']
scaler = StandardScaler()
scaler.fit(features)
standardized_data = scaler.transform(features)
features = standardized_data
x_train,x_test,y_train,y_test = train_test_split(features,tar,test_size=0.2,stratify=tar,random_state=2)
classifier1 = LR.Logistic_Regression(learning_rate=0.01,no_of_iteration=1000)
classifier1.fit(x_train,y_train)
pred = classifier1.predict(x_test)
#training_data_acc = accuracy_score(pred,y_test)
#print(training_data_acc)
#test_data_acc = accuracy_score(y_test,x_test_pred)
#print(test_data_acc)



pickle.dump(classifier,open('diabetes_model.sav','wb'))
pickle.dump(classifier1,open('heart_model.sav','wb'))

