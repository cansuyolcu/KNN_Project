# KNN_Project
A K Nearest Neighbors Project with  a classified data set from a company.
This will be a simple project in order to classify the target class.

## Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

```
## Getting the Data

```python
df = pd.read_csv('KNN_Project_Data')
df.head() 
```
<img src= "https://user-images.githubusercontent.com/66487971/88526861-260cac00-d005-11ea-8f89-9bf9678db35e.png" width = 1000>

## EDA

```python
sns.pairplot(df,hue='TARGET CLASS',palette='coolwarm')
```

<img src= "https://user-images.githubusercontent.com/66487971/88527461-f6aa6f00-d005-11ea-8016-325c24432986.png" width = 3000>

## Standardizing the Variables
Even the variables look close to each other, since they are classified data I choose to standardize them just in case.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()
```

<img src= "https://user-images.githubusercontent.com/66487971/88527982-ac75bd80-d006-11ea-877f-4388a67d48ff.png" width = 900>

## Train Test Split

```python

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'],test_size=0.30)
```

## Using KNN
Trying it for k=1

```python

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

```

## Predictions and Evaluations

```python
pred = knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))

```

<img src= "https://user-images.githubusercontent.com/66487971/88530109-8b629c00-d009-11ea-8d9b-8d07e7ef77a3.png" width = 80>

```python
print(classification_report(y_test,pred))
```

<img src= "https://user-images.githubusercontent.com/66487971/88530289-d4b2eb80-d009-11ea-98e5-e3facf6f00a5.png" width = 400>


## Choosing a K Value

I create a for loop that trains various KNN models with different k values, then keeps track of the error_rate for each of these models with a list.

```python
error_rate = []
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
```
Now I create a plot.

```python
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

```

<img src= "(https://user-images.githubusercontent.com/66487971/88530617-4d19ac80-d00a-11ea-859c-eee99820c689.png" width = 1000>





