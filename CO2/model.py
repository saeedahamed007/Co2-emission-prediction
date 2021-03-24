#Multiple Linear Regression


import pandas as pd
import numpy as np
import pickle


df = pd.read_csv("FuelConsumption.csv")

df.head()



cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)


msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


# plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()



from sklearn import linear_model
# regr = linear_model.LinearRegression()
# x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
# y = np.asanyarray(train[['CO2EMISSIONS']])
# regr.fit (x, y)

# print ('Coefficients: ', regr.coef_)

# y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
# x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
# y = np.asanyarray(test[['CO2EMISSIONS']])
# print("Residual sum of squares: %.2f"
#       % np.mean((y_hat - y) ** 2))


# print('Variance score: %.2f' % regr.score(x, y))



regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x, y)
# print ('Coefficients: ', regr.coef_)
# y_= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
# x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
# y = np.asanyarray(test[['CO2EMISSIONS']])
# print("Residual sum of squares: %.2f"% np.mean((y_ - y) ** 2))
# print('Variance score: %.2f' % regr.score(x, y))




# inputt=[int(x) for x in input().split(' ')]
# final=[np.array(inputt)]

pickle.dump(regr,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))