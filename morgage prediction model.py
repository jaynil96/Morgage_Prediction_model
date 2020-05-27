# -*- coding: utf-8 -*-
"""
Created on Thu May 21 01:01:17 2020

@author: jayni
"""


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/jayni/.spyder-py3/learning/resources/test_Y3wMUE5_7gLdaTN.csv")
print(df.head())


"""
From this we know that Loan_ID, Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area

From this we can assume that the LOAN ID has no effect on the loan
We are trying to build a model that will classify how much loan the user can take
For the sake of simplicity we will be seeing which value effects the amount of money that they can take the most. We will be selecting the top 6
"""
# amount based on gender
df['LoanAmount'].fillna(0, inplace=True)
df['Gender'].fillna("Other", inplace=True)

plt.scatter(df['Gender'], df['LoanAmount'] )
plt.show()

#income vs loan amount
plt.scatter(df["ApplicantIncome"], df['LoanAmount'])
plt.show()
plt.scatter(df["Loan_Amount_Term"], df["LoanAmount"])

"""
We will stick with the Applicant's Income as the main value. 
Now lets build a model
"""
#first off lets split the data
X= df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'Loan_Amount_Term', 'Property_Area']].values
y = df['LoanAmount']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

from sklearn.neighbors import KNeighborsClassifier
k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
print(neigh)

yhat = neigh.predict(X_test)
print(yhat[0:5])

#evaluating
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))