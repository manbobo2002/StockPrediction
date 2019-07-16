# importing the requests library 
import requests
import numpy as np
import datetime
from matplotlib import pyplot
import pandas as pd

#KNN
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#auto arima
from pmdarima.arima import auto_arima

#Prophet
from fbprophet import Prophet

  
# api-endpoint 
URL = "https://www.quandl.com/api/v3/datasets/HKEX/00005.json"

  
# defining a params dict for the parameters to be sent to the API 
now = datetime.datetime.now()
start='2017-01-01'
end=now.strftime("%Y-%m-%d")


# your API key here 
API_KEY = "xxxx"

# data to be sent to api 
data = {'api_key':API_KEY,
        'start_date':start,
        'end_date':end} 
  
# sending get request and saving the response as response object 
r = requests.get(url = URL, data = data) 
  
# extracting data in json format 
res = r.json() 
  
  
  
# printing the output 
jdata=res['dataset']['data']
count=len(res['dataset']['data'])
stockDate=[]
stockPrice=[]
stockHigh=[]
stockLow=[]
stockBid=[]
stockAsk=[]
stockPreviousClose=[]
stockShareVolume=[]
stockTurnover=[]

for i in reversed(range(count)):
    stockDate.append(res['dataset']['data'][i][0])
    stockPrice.append(res['dataset']['data'][i][1])
    stockBid.append(res['dataset']['data'][i][4])
    stockAsk.append(res['dataset']['data'][i][5])
    stockHigh.append(res['dataset']['data'][i][7])
    stockLow.append(res['dataset']['data'][i][8])
    stockPreviousClose.append(res['dataset']['data'][i][9])
    stockShareVolume.append(res['dataset']['data'][i][10])
    stockTurnover.append(res['dataset']['data'][i][11])
    

#setting index as date
stockDate = pd.to_datetime(stockDate,format='%Y-%m-%d')

#Start analysis
new_data = pd.DataFrame(index=range(0,count),columns=['date', 'price', 'bid', 'ask', 'high', 'low', 'previousclose','sharevolume','turnover'])
new_data['date'] = stockDate
new_data['price'] = stockPrice
new_data['bid'] = stockBid
new_data['ask'] = stockAsk
new_data['high'] = stockHigh
new_data['low'] = stockLow
new_data['previousclose'] = stockPreviousClose
new_data['sharevolume'] = stockShareVolume
new_data['turnover'] = stockTurnover
new_data=new_data.dropna()
count=len(new_data)

#splitting into train and validation
Prophet_data=new_data[:];
Prophet_data.rename(columns={'price': 'y', 'date': 'ds'}, inplace=True)
train = new_data[:int(round(count*2/3))]
train_Prophet=Prophet_data[:int(round(count*2/3))]
valid = new_data[int(round(count*2/3)):]
valid_Prophet = Prophet_data[int(round(count*2/3)):]
new_data.shape, train.shape, valid.shape
train['date'].min(), train['date'].max(), valid['date'].min(), valid['date'].max()

#### moving average
preds = []
for i in range(0,count-int(round(count*2/3))):
    a = train['price'][len(train)-(count-int(round(count*2/3)))+i:].sum() + sum(preds)
    b = a/(count-int(round(count*2/3)))
    preds.append(b)
    
#calculate rmse
rms=np.sqrt(np.mean(np.power((np.array(valid['price'])-preds),2)))

#### k-Nearest Neighbours

#scaling data
x_train = train.drop(['price', 'date'], axis=1)
y_train = train['price']
x_valid = valid.drop(['price', 'date'], axis=1)
y_valid = valid['price']
x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)
x_valid_scaled = scaler.fit_transform(x_valid)
x_valid = pd.DataFrame(x_valid_scaled)
#using gridsearch to find the best parameter
params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)

#fit the model and make predictions
model.fit(x_train,y_train)
predsk_Nearest = model.predict(x_valid)

#rmse
rmsk_Nearest=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(predsk_Nearest)),2)))

#### Auto ARIMA
training = train['price']
validation = valid['price']
model = auto_arima(training, start_p=1, start_q=1,max_p=3, max_q=3, m=12,start_P=0, seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)
model.fit(training)
forecast = model.predict(n_periods=len(valid))
forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])
rmsauto_arima=np.sqrt(np.mean(np.power((np.array(valid['price'])-np.array(forecast['Prediction'])),2)))
####Prophet
#preparing data
model = Prophet()
model.fit(train_Prophet)
#predictions
close_prices = model.make_future_dataframe(periods=len(valid_Prophet))
predsprophet = model.predict(close_prices)
#rmse
predsprophet_valid = predsprophet['yhat'][int(round(count*2/3)):]
rmsprophet=np.sqrt(np.mean(np.power((np.array(valid_Prophet['y'])-np.array(predsprophet_valid)),2)))

####plot
pyplot.figure(figsize=(16,8))
#pyplot.plot(x, y)
pyplot.xlabel('date')
pyplot.ylabel('price')
pyplot.plot(new_data['date'], new_data['price'], label="Original")


#moving average
valid['Predictions'] = 0
valid['Predictions'] = preds
pyplot.plot(valid['date'], valid['Predictions'], label="Moving Average")


#KNN
valid['Predictions'] = 0
valid['Predictions'] = predsk_Nearest
pyplot.plot(valid['date'], valid['Predictions'], label="KNN")

#Auto ARIMA
valid['Predictions'] = 0
valid['Predictions'] = forecast
pyplot.plot(valid['date'], valid['Predictions'], label="Auto ARIMA")

#Prophet
valid['Predictions'] = 0
valid['Predictions'] = predsprophet_valid.values
pyplot.plot(valid['date'], valid['Predictions'], label="Prophet")

pyplot.legend(loc='lower left')
