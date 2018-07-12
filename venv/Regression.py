import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
import pickle

style.use('ggplot')


#Get Acxiom stock data
df = quandl.get("WIKI/ACXM")
#Trim down to relevant variables
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
#Create new informative measures
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low'])/df['Adj. Low'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open'] * 100.0

#Final dataset with relevant + new variables.
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
print df.tail()

forecast_col = 'Adj. Close'
#Set missing values to -99999, cannot enter NaN values into a machine learning classifier. Most classivier will recodnise such an obtuse number as an outlier, however.
df.fillna(value =-99999, inplace = True)

#Create Timespan to predict ahead (in this instance, 1% of total records into the future)
forecast_out = int(math.ceil(0.01 *len(df)))

#Create output column, this is the price on the date of the decided timespan into the future (1%)
df['label'] = df[forecast_col].shift(-forecast_out)


#Create input and output numpy arrays
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

#Insert forecast column in dataframe
df['Forecast'] = np.nan


y = np.array(df['label'])

#Train, test and CV
X_Train, X_Test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

#Classifier
clf = LinearRegression(n_jobs = -1)

#Fit the classifier
clf.fit(X_Train, y_train)
confidence = clf.score(X_Test, y_test)
print(confidence)

forcast_set = clf.predict(X_lately)

print(forcast_set, confidence, forecast_out)

#Create forecast column
df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forcast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]


df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

#pickle the classifier

with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)

#open pickled classifier and use:

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

prediction = clf.predict(X_lately)
print(prediction)
