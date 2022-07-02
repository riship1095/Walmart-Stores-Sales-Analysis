import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.image as pltimg
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


file_name='C:\\Users\\DELL\\OneDrive - The University of Texas at Dallas\\Projects\\Walmart BI Project\\Walmart.csv'
data=pd.read_csv(file_name)
data.info()

#There is no null values in the dataframe
#Creating new data variables from the Date variables

data['Year'] = data.Date.str[6:10]
data['Month'] = data.Date.str[3:5]
data['Day'] = data.Date.str[0:2]
data['DateTime'] = pd.to_datetime(data['Date'])  
data['Week_Day'] = data['DateTime'].dt.day_name()

data = data.drop('Date',1)
data = data.drop('DateTime',1)
data = data.drop('Week_Day',1)
data

cols = ['Year','Month','Day']
data[cols] = data[cols].apply(pd.to_numeric, errors='coerce')
data.info()

#Checking for Outliers

fig, axs = plt.subplots(4,figsize=(6,18))
X = data[['Temperature','Fuel_Price','CPI','Unemployment']]
for i,column in enumerate(X):
    sns.boxplot(data[column], ax=axs[i])

# Dropping the Outliers

data = data[(data['Unemployment']<10) & (data['Unemployment']>4.5) & (data['Temperature']>10)]
data

fig, axs = plt.subplots(4,figsize=(6,18))
X = data[['Temperature','Fuel_Price','CPI','Unemployment']]
for i,column in enumerate(X):
    sns.boxplot(data[column], ax=axs[i])

#The Outliers are removed

#Sum of Weekly_Sales for each store, then sorted by total sales

plt.figure(figsize=(15,7))
total_sales_for_each_store = data.groupby('Store')['Weekly_Sales'].sum().sort_values()
clrs = ['lightsteelblue' if ((x < max(total_sales_for_each_store)) and (x > min(total_sales_for_each_store))) else 'midnightblue' for x in total_sales_for_each_store]
total_sales_for_each_store.plot(kind='bar',color=clrs);
plt.xlabel('Store')
plt.ylabel('Total Sales(in 10 millions)')
plt.title('Total sales for each store')

# Store 20 has the highest total sales around \\$30 millions while Store 33 has the lowest total sales lesser than \\$5millions

# Finding Store that has the maximum standard deviation of weekly sales of all the stores. 

data_std = pd.DataFrame(data.groupby('Store')['Weekly_Sales'].std().sort_values(ascending=False))
data_std.head(1)
print('The store that has maximum standard deviation is',(data_std.head(1).index[0]))

# Effect of Fuel Price on weekly sales of Store #14

plt.figure(figsize=(15,7))
x= data['Fuel_Price'][data['Store']==14]
y= data[data['Store'] == data_std.head(1).index[0]]['Weekly_Sales']
plt.scatter(x,y)
z= np.polyfit(x,y,1)
p= np.poly1d(z)
plt.plot(x,p(x))
plt.xlabel('Fuel Price')
plt.ylabel('Total Sales(in 10 millions)')
plt.title('Effect of fuel price on sales of Store #'+ str(data_std.head(1).index[0]));

# There is a gradual decrease in weekly sales of Store #14 due to increase in fuel price. 

# Effect of Unemployment rate on Weekly Sales

plt.figure(figsize=(15,7))
x1= np.array(data['Unemployment'])
y1= np.array(data['Weekly_Sales'])
plt.scatter(x1,y1)
z= np.polyfit(x1,y1,1)
p= np.poly1d(z)
plt.plot(x1,p(x1))
plt.xlabel('Unemployment Rate')
plt.ylabel('Total Sales(in 10 millions)')
plt.title('Overall effect of Unemployment rate on Weekly sales')

# There does not seem to be any effect of increasing unemployment on the weekly sales of Walmart stores

# Effect of CPI on Weekly Sales

plt.figure(figsize=(15,7))
x1= np.array(data['CPI'])
y1= np.array(data['Weekly_Sales'])
plt.scatter(x1,y1)
z= np.polyfit(x1,y1,1)
p= np.poly1d(z)
plt.plot(x1,p(x1))
plt.xlabel('CPI')
plt.ylabel('Total Sales(in 10 millions)')
plt.title('Overall effect of Consumer Price Index on Weekly sales')

# There seem to be slight decrease in weekly sales of increasing of CPI

# Monthly Sales for each year

plt.bar(data[data.Year==2010]["Month"],data[data.Year==2010]["Weekly_Sales"])
plt.xlabel("months")
plt.ylabel("Weekly Sales")
plt.title("Monthly view of sales in 2010")
plt.show()
plt.bar(data[data.Year==2011]["Month"],data[data.Year==2011]["Weekly_Sales"])
plt.xlabel("months")
plt.ylabel("Weekly Sales")
plt.title("Monthly view of sales in 2011")
plt.show()
plt.bar(data[data.Year==2012]["Month"],data[data.Year==2012]["Weekly_Sales"])
plt.xlabel("months")
plt.ylabel("Weekly Sales")
plt.title("Monthly view of sales in 2012")
plt.show()

# Import sklearn 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression

# Select independent variables and target variables
X = data[['Store','Fuel_Price','CPI','Unemployment','Day','Month','Year']]
y = data['Weekly_Sales']

# Split data to train and test (0.80:0.20)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# Linear Regression model
print('Linear Regression:')
print()
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print('Accuracy:',reg.score(X_train, y_train)*100)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

sns.scatterplot(y_pred, y_test);

# Random Forest Regressor
print('Random Forest Regressor:')
print()
rfr = RandomForestRegressor(n_estimators = 400,max_depth=15,n_jobs=5)        
rfr.fit(X_train,y_train)
y_pred=rfr.predict(X_test)
print('Accuracy:',rfr.score(X_test, y_test)*100)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

sns.scatterplot(y_pred, y_test);


