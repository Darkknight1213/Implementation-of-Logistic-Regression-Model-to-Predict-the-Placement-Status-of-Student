# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset, drop unnecessary columns, and encode categorical variables.
2. Define the features (X) and target variable (y).
3. Split the data into training and testing sets.
4. Train the logistic regression model, make predictions, and evaluate using accuracy and other

## Program:
```python
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Mohamed Riyaz Ahamed 
RegisterNumber: 212224240092
*/

import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
```
![image](https://github.com/user-attachments/assets/c8f66116-9a75-4421-b9bb-19172f1a604f)

```python
data1=data.copy()
data1=data1.drop(["sl_no", "salary"], axis=1) 
data1.head()
```
![image](https://github.com/user-attachments/assets/f9b92cf9-0ab1-4980-a436-1a385431eb04)

```python
data1.duplicated().sum()
```
![image](https://github.com/user-attachments/assets/c10fba21-bf5d-4ba2-95cb-ecf5f286590b)

```python
data1.isnull().sum()
```
![image](https://github.com/user-attachments/assets/d60bcc32-b94d-4c9e-84cd-8ca5af3e622b)


```python
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])

data1["hsc_b"]=le.fit_transform(data1["hsc_b"]) 
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])

data1["degree_t"]=le.fit_transform(data1["degree_t"]) 
data1["workex"]=le.fit_transform(data1["workex"])

data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])

data1
```
![image](https://github.com/user-attachments/assets/8fcba144-2b59-40f9-8b87-fdf29b1de939)


```python
x=data1.iloc[:,:-1]
x
```
![image](https://github.com/user-attachments/assets/a9552177-3796-4d90-9f6b-696432db9913)

```python
y=data1["status"]
y
```
![image](https://github.com/user-attachments/assets/a5a1aa7d-d5e6-47dc-a9aa-65d30b507f8c)

```python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression (solver="liblinear") 
lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)
y_pred
lr.score(x_test,y_test)
```
![image](https://github.com/user-attachments/assets/c5026ba0-1f31-4772-ba9c-49a80fd7b8d9)
![image](https://github.com/user-attachments/assets/4270193e-d13c-48ef-ba17-9484557739d8)



```python
from sklearn.metrics import confusion_matrix
confusion=(y_test,y_pred) 
confusion
```
![image](https://github.com/user-attachments/assets/00399d2d-65a1-4c97-82ff-03f430b15617)


```python
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,0,1,1,90,1,0,85,1,85]])
```
out
## Output:
![image](https://github.com/user-attachments/assets/36a914d8-227f-4625-a2f4-d6f4d8b53e00)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
