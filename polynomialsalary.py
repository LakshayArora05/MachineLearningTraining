#PROJECT-2:POLYNOMIAL REGRESSION OF DEGREE 3 - SALARY VS YEARS OF EXPERIENCE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
salary=pd.read_csv(r"Desktop\Employee_Salary.csv")
salary.head(5)
X=salary[["Years of Experience"]]
Y=salary["Salary"]
X_train=X
Y_train=Y
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_regressor=PolynomialFeatures(degree=3)
X_columns=poly_regressor.fit_transform(X_train)
regressor=LinearRegression()
regressor.fit(X_columns,Y_train)
print("Model's coefficient are:",regressor.coef_)
y_predict=regressor.predict(poly_regressor.fit_transform(X_train))
plt.scatter(X_train,Y_train)
plt.xlabel("YEARS")
plt.ylabel("salary")
plt.plot(X_train,y_predict,color='red')
