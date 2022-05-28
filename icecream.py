#PROJECT 1: ICECREAM BUSINESS REVENUE PREDICTION USING SIMPLE LINEAR REGRESSION
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import dataset
IceCream=pd.read_csv(r'C:\Users\Lakshay\anaconda\IceCreamData.csv')

#check if imported

IceCream.head(10)
IceCream.tail(10)
IceCream.describe()
IceCream.info()

#visualize dataset
sns.jointplot(x='Temperature', y='Revenue', data = IceCream, color = 'blue')
sns.pairplot(IceCream)
sns.lmplot(x='Temperature',y='Revenue', data=IceCream)

#create testing and training set
X=IceCream[['Temperature']]
y=IceCream['Revenue']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)
X_train
X_test
y_train
y_test
X_train.shape
X_test.shape
y_train.shape
y_test.shape



#train dataset
from sklearn.linear_model import LinearRegression
regressor=LinearRegression(fit_intercept=True)
regressor.fit(X_train,y_train)
print("Linear regression coefficient m is:", regressor.coef_)
print("Linear regression coefficient c is:", regressor.intercept_)


#test dataset
y_predict= regressor.predict(X_test)
plt.scatter(X_train,y_train,color="red")
plt.title("REVENUE VS TEMPERATURE GRAPH FOR ICECREAM BUSINESS")
plt.plot(X_train,regressor.predict(X_train), color='blue')

y_predict= regressor.predict(X_test)
plt.scatter(X_test,y_test,color="red")
plt.title("REVENUE VS TEMPERATURE GRAPH FOR ICECREAM BUSINESS")
plt.plot(X_test,regressor.predict(X_test), color='blue')


#predict for any value
T = 40
T = np.array(T).reshape(1, -1)
y_predict = regressor.predict(T)
y_predict
