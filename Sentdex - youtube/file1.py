import datetime
import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing,model_selection,svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
# print(df.head())

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close']*100
df['PCT_Change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100
df = df[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume']]
# print(df.head())

forecast_col = 'Adj. Close'
df.fillna(-99999,inplace=True)
forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)
# df['label'] = df[forecast_col].shift(-forecast_out)
# print(df.head())
# df.dropna(inplace=True)
# print(df.tail())
# print(df.head())

df['label'] = df[forecast_col].shift(-forecast_out)
X = np.array(df.drop(['label'], axis=1))
# y = np.array(df['label'])

X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]
df.dropna(inplace=True)
y = np.array(df['label'])

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.2)
# clf = LinearRegression()
# clf = svm.SVR()
# clf = svm.SVR(kernel = 'poly')
clf = LinearRegression(n_jobs=10)
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)
# print(accuracy)

forecast_set = clf.predict(X_lately)
print(forecast_set,accuracy,forecast_out)

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
print(df.head())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()