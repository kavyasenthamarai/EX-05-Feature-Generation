# EX-05-Feature-Generation


## AIM
To read the given data and perform Feature Generation process and save the data to a file. 

# Explanation
Feature Generation (also known as feature construction, feature extraction or feature engineering) is the process of transforming features into new features that better relate to the target.
 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Generation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE
```
DATA.CSV
import pandas as pd
df=pd.read_csv("data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
import category_encoders as ce
be=ce.BinaryEncoder()
ohe=OneHotEncoder(sparse=False)
le=LabelEncoder()
oe=OrdinalEncoder()


df1["City"] = ohe.fit_transform(df1[["City"]])

temp=['Cold','Warm','Hot','Very Hot']
oe1=OrdinalEncoder(categories=[temp])
df1['Ord_1'] = oe1.fit_transform(df1[["Ord_1"]])

edu=['High School','Diploma','Bachelors','Masters','PhD']
oe2=OrdinalEncoder(categories=[edu])
df1['Ord_2']= oe2.fit_transform(df1[["Ord_2"]])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df5
ENCODING.CSV
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
le=LabelEncoder()
oe=OrdinalEncoder()

df1["nom_0"] = oe.fit_transform(df1[["nom_0"]])
temp=['Cold','Warm','Hot']
oe2=OrdinalEncoder(categories=[temp])
df1['ord_2'] = oe2.fit_transform(df1[['ord_2']])

df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df0=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df0

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df2=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df2

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df3=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df3

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df4=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df4
# TITANIC.CSV

import pandas as pd
df=pd.read_csv("titanic_dataset.csv")
df

#removing unwanted data
df.drop("Name",axis=1,inplace=True)
df.drop("Ticket",axis=1,inplace=True)
df.drop("Cabin",axis=1,inplace=True)

#data cleaning
df.isnull().sum()

df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

df.isnull().sum()

df

#feature encoding
from category_encoders import BinaryEncoder
be=BinaryEncoder()
df["Sex"]=be.fit_transform(df[["Sex"]])
ndf=be.fit_transform(df["Sex"])
ndf

df1=df.copy()
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
embark=['S','C','Q']
e1=OrdinalEncoder(categories=[embark])
df1['Embarked'] = e1.fit_transform(df[['Embarked']])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df5
```
OUPUT:
DATA CSV
# Initial Dataset:

![image](https://user-images.githubusercontent.com/118668727/237002307-b2a8e3c0-f0b9-4bbb-8f4e-5ad0a2bfc0cb.png)
# Binary Encoding:
![image](https://user-images.githubusercontent.com/118668727/237002381-721e542b-0406-451f-8627-12ea44aee672.png)
# Encoded Dataset:
![image](https://user-images.githubusercontent.com/118668727/237002432-cbc84dd8-a6f7-4c38-8700-a5ac5d0ba77e.png)
# Data Scaling using MinMaxScaler:
![image](https://user-images.githubusercontent.com/118668727/237002479-768a208d-ded1-4886-9ad2-cea7c4c7d532.png)
# Data Scaling using StandardScaler:
![image](https://user-images.githubusercontent.com/118668727/237002788-143a75e5-28c7-402d-86c5-b137729f8938.png)
# Data Scaling using MaxAbsScaler:
![image](https://user-images.githubusercontent.com/118668727/237002845-562b3cfb-477b-4cd8-bec2-80e08dea3b2c.png)
# Encoding.csv :
# Initial Dataset:
![image](https://user-images.githubusercontent.com/118668727/237002924-dfd0f926-2433-4b66-bc20-5fda988fc33e.png)
# Binary Encoding:
![image](https://user-images.githubusercontent.com/118668727/237003118-30c00540-e498-49cd-a0c0-e42bc4a61853.png)
# Encoded Dataset:
![image](https://user-images.githubusercontent.com/118668727/237003149-17ffb3a5-0f41-403c-ac51-c01c8af57e70.png)
# Data Scaling using MinMaxScaler:
![image](https://user-images.githubusercontent.com/118668727/237003285-f5e540e9-7773-41f0-8353-4bea90970036.png)
# #Titanic.csv :
# Initial Dataset:
![image](https://user-images.githubusercontent.com/118668727/237003417-111d443f-df24-4a8a-bbcd-c39a4426b7bc.png)
# Data cleaning before encoding:
![image](https://user-images.githubusercontent.com/118668727/237003510-e35f1a19-6d50-40e7-94c6-6ed5a9202501.png)
# Cleaned Dataset:
![image](https://user-images.githubusercontent.com/118668727/237003630-c40f0859-c0aa-42b7-98e1-34f7ea50e80e.png)
# Binary Encoding:
![image](https://user-images.githubusercontent.com/118668727/237003669-a4f31888-0797-42cc-ad7a-b0bb93dfe0d5.png)
# Encoded Dataset:
![image](https://user-images.githubusercontent.com/118668727/237003985-3c2aa245-3108-4107-8d24-f5cc8c8a7106.png)

# Data Scaling using MinMaxScaler:
![image](https://user-images.githubusercontent.com/118668727/237004055-6edabe1b-06fc-42ab-ab1e-f32723ecee9c.png)
# Data Scaling using StandardScaler:
![image](https://user-images.githubusercontent.com/118668727/237004262-a496ffaa-a013-4ebe-bd73-118a0fc01308.png)
# Data Scaling using MaxAbsScaler:
![image](https://user-images.githubusercontent.com/118668727/237004319-19782093-9dd9-44b7-9482-370344fa40af.png)
# Data Scaling using RobustScaler:
![image](https://user-images.githubusercontent.com/118668727/237004382-e32c434a-ba89-4c05-ab1b-c23c883bfbbc.png)
RESULT:
Feature Generation process and Feature Scaling process is applied to the given data frames sucessfully.



