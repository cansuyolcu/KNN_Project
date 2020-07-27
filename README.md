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

<img src= "https://user-images.githubusercontent.com/66487971/88527982-ac75bd80-d006-11ea-877f-4388a67d48ff.png" width = 1000>


