# Ex-06-Feature-Transformation
# AIM:
To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM:
# STEP 1:
Read the given Data

# STEP 2:
Clean the Data Set using Data Cleaning Process

# STEP 3:
Apply Feature Transformation techniques to all the features of the data set

# STEP 4:
Save the data to the file

# PROGRAM:
Name : s.thirisha
Register Number : 212222230160
```python 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
df=pd.read_csv("/content/Data_to_Transform.csv")
df
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()
sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()
sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()
sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()
df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df['HighlyNegativeSkew']=np.log(df.HighlyNegativeSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df['ModeratePositiveSkew_1'], parameters=stats.yeojohnson(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()
df['ModerateNegativeSkew_1'], parameters=stats.yeojohnson(df.ModerateNegativeSkew)
sm.qqplot(df.ModerateNegativeSkew_1,fit=True,line='45')
plt.show()
from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()
from sklearn.preprocessing import QuantileTransformer
qt= QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()
```
# OUTPUT:
![image](https://user-images.githubusercontent.com/120380280/233552084-c7a1f157-123b-408a-a98f-8415775f10d9.png)

![image](https://user-images.githubusercontent.com/120380280/233552133-edc9e9d6-a645-4156-8c1c-d27cd157d8b5.png)

![image](https://user-images.githubusercontent.com/120380280/233552153-02ed2454-706d-4f5b-b0e1-39314b1936e7.png)

![image](https://user-images.githubusercontent.com/120380280/233552235-83d76c05-6c3b-4cc0-805f-2cd3dcffbcb2.png)

![image](https://user-images.githubusercontent.com/120380280/233552265-8d7f1bad-f86d-4177-abec-657eb78f84ba.png)

![image](https://user-images.githubusercontent.com/120380280/233552290-0f935703-8627-4423-9dd1-c74f4470ef22.png)

![image](https://user-images.githubusercontent.com/120380280/233552307-f0999d4f-e997-42b9-804f-d9104926aaea.png)

![image](https://user-images.githubusercontent.com/120380280/233552323-e78cbdd0-2961-44bf-999d-92f4625d1167.png)

![image](https://user-images.githubusercontent.com/120380280/233552348-3a94bd68-a26f-4f27-823a-fcc42d9df1f5.png)

![image](https://user-images.githubusercontent.com/120380280/233552368-ba247ef4-d1d2-44af-b4b1-5300e0aa4294.png)

![image](https://user-images.githubusercontent.com/120380280/233552376-f8c378ae-5d86-4fb7-9183-8b8c1046310e.png)

# RESULT:
Thus the Feature Transformation for the given datasets had been executed successfully
