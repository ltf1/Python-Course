import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
import seaborn as sns
import matplotlib.pyplot as plt
from LinearRegressionClass import *


from sklearn.datasets import load_boston
boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target
df = boston
boston["Intercept"] = 1  # creating a vector of ones
boston = boston[["Intercept", "LSTAT", "RM", "CRIM", "DIS", "MEDV"]]
boston.isnull().sum()



#plotting dependent variablespython DataAndApplicationOfRegression.py
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(boston['MEDV'], bins=30)
plt.show()

X = pd.DataFrame(np.c_[boston['Intercept'], boston['RM'], boston['LSTAT'], boston["CRIM"], boston["DIS"]], columns = ["Intercept",'Average_Rooms', 'Lower_Status', "Crime_Rate", "Distance_Emp_Centers"])
y = boston['MEDV']
y
X = boston[["Intercept", "LSTAT", "RM", "CRIM", "DIS",]].values
y = boston[["MEDV"]].values


#model fitting

model = LinearRegressionClass()
model.fit(X,y)
model.coefficients


# StandardErrors

model.StandardErrors(cross_tab)
model.r_square


# verify this with SM OLS below

import statsmodels.api as sm

regressor = sm.OLS(y, X).fit()
print(regressor.summary())
