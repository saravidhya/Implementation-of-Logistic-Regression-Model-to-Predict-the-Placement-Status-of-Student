# EXPERIMENT NO: 5
# Implementation of Logistic Regression Model to Predict the Placement Status of Student
### NAME : VIDHIYA LAKSHMI S
### REG NO: 212223230238
## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values. 
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.

## Program & Output:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: VIDHIYA LAKSHMI S
RegisterNumber:  212223230238
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
dataset = pd.read_csv('Placement_Data_Full_Class (1).csv')
dataset
```
![image](https://github.com/user-attachments/assets/bddcac13-8d1b-4923-b675-b2bb8f6367f5)
```
dataset.info()
```
![image](https://github.com/user-attachments/assets/7150dcc0-3327-4e53-a277-abeaa862a81e)
```
dataset.drop('sl_no',axis=1)
dataset.info()
```
![image](https://github.com/user-attachments/assets/b15ee8f8-6797-4c94-8d2e-70e9eb421ab4)
```
dataset["gender"]= dataset["gender"].astype('category')
dataset["ssc_b"]= dataset["ssc_b"].astype('category')
dataset["hsc_b"]= dataset["hsc_b"].astype('category')
dataset["hsc_s"]= dataset["hsc_s"].astype('category')
dataset["degree_t"]= dataset["degree_t"].astype('category')
dataset["workex"]= dataset["workex"].astype('category')
dataset["specialisation"]= dataset["specialisation"].astype('category')
dataset["status"]= dataset["status"].astype('category')
dataset.dtypes
dataset.info()
```
![image](https://github.com/user-attachments/assets/0083473d-e62b-4ccb-81cd-418f172f0616)
```
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset
```
![image](https://github.com/user-attachments/assets/9d0b9898-f33d-4e8e-97bb-8e94b8691e1b)
```
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
clf = LogisticRegression()
clf.fit(x_train,y_train)
```
![image](https://github.com/user-attachments/assets/9ab35a66-d742-4b6d-a435-a7ba5cc90284)
```
X_train.shape
```
![image](https://github.com/user-attachments/assets/eda27cef-2538-4d4c-b840-9610a844d420)
```
X_test.shape
```
![image](https://github.com/user-attachments/assets/211c7b2c-d35e-4927-9722-f9f1fa22bd21)
```
Y_train.shape
```
![image](https://github.com/user-attachments/assets/a8ae771e-b9c1-469c-9b1a-ac75e91f1878)
```
Y_test.shape
```
![image](https://github.com/user-attachments/assets/ccfbd911-c0cf-4d25-90b0-0cfc0bf6e4fd)
```
y_pred=clf.predict(x_test)
y_pred
```
![image](https://github.com/user-attachments/assets/b6b27f37-2df4-4d8d-b245-b43334a36562)
```
from sklearn.metrics import confusion_matrix, accuracy_score
cf = confusion_matrix(y_test, y_pred)
print(cf)
```
![image](https://github.com/user-attachments/assets/990e0084-8827-495f-b8bd-bd532714a250)
```
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
```
![image](https://github.com/user-attachments/assets/9e67e70b-e9fe-416e-9c4a-9889f08e0b81)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
