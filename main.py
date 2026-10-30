from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, root_mean_squared_error
import pandas as pd
cutDict = {
    'Fair':0,
    'Good':1,
    'Very Good':2,
    'Premium':3,
    'Ideal':4
}
colorDict = {
    'J':0,
    'I':1,
    'H':2,
    'G':3,
    'F':4,
    'E':5,
    'D':6
}
clarityDict = {
    'I3':0,
    'I2':1,
    'I1':2,
    'SI2':3,
    'SI1':4,
    'VS2':5,
    'VS1':6,
    'VVS2':7,
    'VVS1':8,
    'IF':9,
    'FL':10
}
data = pd.read_csv("archive/cubic_zirconia.csv",index_col=0)
X = data
X = X.drop(columns='price')
X['cut'] = X['cut'].map(cutDict)
X['color'] = X['color'].map(colorDict)
X['clarity'] = X['clarity'].map(clarityDict)
Y = data['price']
model = RandomForestRegressor(n_estimators=500,random_state=42)
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
model.fit(x_train,y_train)
pred = model.predict(x_test)
# FIX CONTINUE CONFIG ERROR
mae = mean_absolute_error(y_test,pred)
mse = mean_squared_error(y_test,pred)
rmse = root_mean_squared_error(y_test,pred)
r2 = r2_score(y_test,pred)
mape = mean_absolute_percentage_error(y_test,pred)
print("MAE: " + str(mae))
print("MSE: " + str(mse))
print("RMSE: " + str(rmse))
print("R2: " + str(r2))
print("MAPE: " + str(mape))