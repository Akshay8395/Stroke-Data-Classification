# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 00:43:34 2020

@author: Akshay
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import metrics
from yellowbrick.classifier import ClassificationReport
from sklearn.svm import SVC   
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from warnings import simplefilter
from sklearn.tree import DecisionTreeClassifier
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
#read data from CSV file
stroke_data = pd.read_csv("STROKE.csv")
# Conversting catogerical data to numerical values to use it of different types classifires
stroke_data = stroke_data.replace(to_replace = ['Male','Female','Other'],value = [1,0,2])
stroke_data = stroke_data.replace(to_replace = ['Yes','No'],value = [1,0])
stroke_data = stroke_data.replace(to_replace = ['children','Govt_job','Never_worked','Private','Self-employed'],value = [0,1,2,3,4])
stroke_data = stroke_data.replace(to_replace = ['Rural','Urban'],value = [0,1])

#Considering important factors in data set as data and Stroke as traget  variable
X = stroke_data[['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi']]
Y = stroke_data['stroke']


#split data set into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.40, random_state = 10)


print("Naive-Bayes")
#Fit the model 
gnb = GaussianNB().fit(X_train,Y_train)

#
y_pred = gnb.predict(X_test)

Accuracy = accuracy_score(Y_test,y_pred)
print("Accuracy using Naive-Bayes is ",round((Accuracy)*100,4),"%")

visualizer = ClassificationReport(gnb, classes=['1','0'])
visualizer.fit(X_train, Y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, Y_test)  # Evaluate the model on the test data
g = visualizer.poof() 

print("Logistic Regression")
#Fit the model
LR = LogisticRegression().fit(X_train,Y_train)

#
y_pred = LR.predict(X_test)

Accuracy = accuracy_score(Y_test,y_pred)
print("Accuracy using Logistic regression is ",round((Accuracy)*100,4),"%")

visualizer = ClassificationReport(LR, classes=['1','0'])
visualizer.fit(X_train, Y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, Y_test)  # Evaluate the model on the test data
g = visualizer.poof() 


print("AdaBoost")
#Fit the model 
ABC = AdaBoostClassifier(n_estimators=50,learning_rate=1).fit(X_train,Y_train)

#
y_pred = ABC.predict(X_test)

Accuracy = accuracy_score(Y_test,y_pred)
print("Accuracy using AdaBoost is ",round((Accuracy)*100,4),"%")

visualizer = ClassificationReport(ABC, classes=['1','0'])
visualizer.fit(X_train, Y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, Y_test)  # Evaluate the model on the test data
g = visualizer.poof()

print("Random Forest")
#Fit the model 
RF = RandomForestClassifier().fit(X_train,Y_train)


y_pred = RF.predict(X_test)

Accuracy = accuracy_score(Y_test,y_pred)
print("Accuracy using Random Forest is ",round((Accuracy)*100,4),"%")

visualizer = ClassificationReport(RF, classes=['1','0'])
visualizer.fit(X_train, Y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, Y_test)  # Evaluate the model on the test data
g = visualizer.poof() 

print("Decision Tree")
#Fit the model 
DT =DecisionTreeClassifier(criterion="entropy").fit(X_train,Y_train)

#
y_pred = DT.predict(X_test)

Accuracy = accuracy_score(Y_test,y_pred)
print("Accuracy using Decision Tree is ",round((Accuracy)*100,4),"%")

visualizer = ClassificationReport(DT, classes=['1','0'])
visualizer.fit(X_train, Y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, Y_test)  # Evaluate the model on the test data
g = visualizer.poof() 


print("Support Vector Machine")
#Fit the model
SVM = SVC(kernel='linear').fit(X_train,Y_train)

#
y_pred = SVM.predict(X_test)

Accuracy = accuracy_score(Y_test,y_pred)
print("Accuracy using linear SVM is ",round((Accuracy)*100,4),"%")

visualizer = ClassificationReport(SVM, classes=['1','0'])
visualizer.fit(X_train, Y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, Y_test)  # Evaluate the model on the test data
g = visualizer.poof() 
