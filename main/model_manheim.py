
"""### Model Developing"""

# Importing Modules

import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

#libraries for preprocessing
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#libraries for evaluation
from sklearn.metrics import mean_squared_log_error,r2_score,mean_squared_error
from sklearn.model_selection import train_test_split


#libraries for models
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoCV,RidgeCV
#from yellowbrick.regressor import AlphaSelection

from sklearn.linear_model import Lasso

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from sklearn.ensemble import AdaBoostRegressor

import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

#Loading Dataframe

df=pd.read_csv("data/vehicles_Manheim_Final.csv")
#df=df.drop('Unnamed: 0',axis=1)

df2=df.copy()
#df.head()

#defining numerical and categorical values
num_col=['Year', 'Odometer']
cat_cols=['Make','Model','Color','Trans','4x4','Top']

# Label Encoding
le=preprocessing.LabelEncoder()
df[cat_cols]=df[cat_cols].apply(le.fit_transform)
df.head(2)

# Scaling numerical data
norm = StandardScaler()
df['Price'] = np.log(df['Price'])
df['Odometer'] = norm.fit_transform(np.array(df['Odometer']).reshape(-1,1))
df['Year'] = norm.fit_transform(np.array(df['Year']).reshape(-1,1))
df['Model'] = norm.fit_transform(np.array(df['Model']).reshape(-1,1))

#scaling target variable
q1,q3=(df['Price'].quantile([0.25,0.75]))
o1=q1-1.5*(q3-q1)
o2=q3+1.5*(q3-q1)
df=df[(df.Price>=o1) & (df.Price<=o2)]

#df.head(2)

# Function to split dataset int training and test
def trainingData(df,n):
    X = df.iloc[:,n]
    y = df.iloc[:,-1:].values.T
    y=y[0]
    X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.9,test_size=0.1,random_state=0)
    return (X_train,X_test,y_train,y_test)

X_train,X_test,y_train,y_test=trainingData(df,list(range(len(list(df.columns))-1)))

# Some of models will predict neg values so this function will remove that values

def remove_neg(y_test,y_pred):
    ind=[index for index in range(len(y_pred)) if(y_pred[index]>0)]
    y_pred=y_pred[ind]
    y_test=y_test[ind]
    y_pred[y_pred<0]
    return (y_test,y_pred)

#function for evaluation of model
def result(y_test,y_pred):
    r=[]
    r.append(mean_squared_log_error(y_test, y_pred))
    r.append(np.sqrt(r[0]))
    r.append(r2_score(y_test,y_pred))
    r.append(round(r2_score(y_test,y_pred)*100,4))
    return (r)

#dataframe that store the performance of each model
accu=pd.DataFrame(index=['MSLE', 'Root MSLE', 'R2 Score','Accuracy(%)'])

"""### Linear Regression"""

# Fitting model
LR=LinearRegression()
LR.fit(X_train,y_train)
y_pred=LR.predict(X_test)

# Calculating error/accuracy
y_test_1,y_pred_1=remove_neg(y_test,y_pred)
r1_lr=result(y_test_1,y_pred_1)
print("---------- Linear Regression ----------")
print('Coefficients: \n', LR.coef_)
print("MSLE : {}".format(r1_lr[0]))
print("Root MSLE : {}".format(r1_lr[1]))
print("R2 Score : {} or {}%".format(r1_lr[2],r1_lr[3]))
accu['Linear Regression']=r1_lr

# Ploting feature importance graph

coef = pd.Series(LR.coef_, index = X_train.columns)
imp_coef = coef.sort_values()
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Linear Regression Model")
plt.savefig('plots/Linear-Regression-Feature-Importance.jpg')
#plt.show()

# Visualization of true value and predicted
df_check = pd.DataFrame({'Actual': y_test_1, 'Predicted': y_pred_1})
df_check = df_check.sample(20)
df_check.plot(kind='bar',figsize=(12,8))
plt.grid(which='major', linestyle='-', linewidth='0.1', color='Green')
plt.title('Performance of Linear Regression')
plt.savefig('plots/Linear-Regression-Performance')
#plt.show()



"""### Random Forest"""

# Model
RFR = RandomForestRegressor(n_estimators=180,random_state=0, min_samples_leaf=1, max_features=0.5, n_jobs=-1, oob_score=True)
RFR.fit(X_train,y_train)
y_pred = RFR.predict(X_test)

# Results
r5_rf=result(y_test,y_pred)
print("---------- Random Forest ----------")
print("MSLE : {}".format(r5_rf[0]))
print("Root MSLE : {}".format(r5_rf[1]))
print("R2 Score : {} or {}%".format(r5_rf[2],r5_rf[3]))
accu['RandomForest Regressor']=r5_rf

# Visualizing Predicted and Real Values
df_check = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_check = df_check.head(25)
#round(df_check,2)
df_check.plot(kind='bar',figsize=(12,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.title('Performance of Random Forest')
plt.ylabel('Mean Squared Log Error')
plt.savefig('plots/Random-Forest-Performance.jpg')
#plt.show()

"""### XGBOOST"""

# Model implementation and fitting data
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.4,
                max_depth = 24, alpha = 5, n_estimators = 200)
xg_reg.fit(X_train,y_train)
y_pred = xg_reg.predict(X_test)

# Model evaluation
y_test_1,y_pred_1=remove_neg(y_test,y_pred)
r8_xg=result(y_test_1,y_pred_1)
print("---------- XGBOOST ----------")
print("MSLE : {}".format(r8_xg[0]))
print("Root MSLE : {}".format(r8_xg[1]))
print("R2 Score : {} or {}%".format(r8_xg[2],r8_xg[3]))

# Visualizing Predicted and Real Values
df_check = pd.DataFrame({'Actual': y_test_1, 'Predicted': y_pred_1})
df_check = df_check.head(25)
#round(df_check,2)
df_check.plot(kind='bar',figsize=(12,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.title('Performance of XGBOOST')
plt.ylabel('Mean Squared Log Error')
plt.savefig('plots/XGBOOST-Performance.jpg')
#plt.show()


"""### Predict"""

#Loading Dataframe
df_pre=pd.read_csv("predictions_to_make.csv")
df_predict = df_pre

# Label Encoding
le=preprocessing.LabelEncoder()
df_predict[cat_cols]=df_predict[cat_cols].apply(le.fit_transform)

#Formatting it for model
n = list(range(len(list(df_predict.columns))-1))
X_to_pred = df_predict.iloc[:,n]
y_actual = df_predict.iloc[:,-1:].values.T
y_actual=y_actual[0]

#Predictions
y_predicted = xg_reg.predict(X_to_pred)

arr = np.exp(y_predicted)
predicted = arr.tolist()
df_pre['Predicted Price'] = predicted
#df_predict.head

# Result to csv file
df_pre.to_csv('predictions_to_make.csv', index=False)
print(df_pre)