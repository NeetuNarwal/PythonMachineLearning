# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 09:26:30 2019

@author: neetu
"""
import os
os.chdir("e:\pythondatascience")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
car=pd.read_csv("cars_sampled.csv")
car.drop_duplicates(keep='first',inplace=True)
car.drop(columns=['name','seller','brand','dateCrawled','postalCode','lastSeen'],inplace=True)
car=car.apply(lambda x: x.fillna(x.mean()) if x.dtype=='float64' else x.fillna(x.value_counts().index[0]))
car=car[(car.price>100) & (car.price<150000)]
cars=car.dropna(axis=0)
cars=pd.get_dummies(cars,drop_first=True)
x=cars.drop('price',axis=1,inplace=False)
y=cars['price']

y=np.log(y)
trainx,testx,trainy,testy=train_test_split(x,y,test_size=0.3,random_state=1)
basepred=np.mean(trainy)
basepred=np.repeat(basepred,len(testy))
brmse=np.sqrt(mean_squared_error(testy,basepred))
print(brmse)
lin=LinearRegression(fit_intercept=True)
model=lin.fit(trainx,trainy)
pre=lin.predict(testx)
rmse=np.sqrt(mean_squared_error(testy,pre))
print(rmse)
print(model.score(trainx,trainy))
print(model.score(testx,testy))
residual=testy-pre
sns.regplot(residual,pre)

################ RandomForestRegressor
rm=RandomForestRegressor(n_estimators=100,max_depth=100,max_features='auto',min_samples_leaf=4,random_state=1)
model1=rm.fit(trainx,trainy)
pred=rm.predict(testx)
rmse1=np.sqrt(mean_squared_error(testy,pred))
print(rmse1)
print(model1.score(trainx,trainy))
print(model1.score(testx,testy))

#new data
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
#q1
ndiamond=pd.get_dummies(diamond,drop_first=True)
#q2
from sklearn.model_selection import train_test_split
ndiamond.dropna(axis=0)
x=ndiamond.drop('price',axis=1,inplace=False)
y=ndiamond['price']
trainx,testx,trainy,testy=train_test_split(x,y,test_size=0.3,random_state=0)
print(trainx.shape,testx.shape,trainy.shape,testy.shape)
