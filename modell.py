# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 13:08:55 2021

@author: Anunay
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 13:08:55 2021

@author: Anunay
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv(r'C:\Users\Anunay\Desktop\mling\artistrevenue\spotify.csv')

dataset['active(months)'].fillna(0, inplace=True)

dataset['followers'].fillna(dataset['followers'].mean(), inplace=True)

X = dataset.iloc[:, :3]




y = dataset.iloc[:, -1]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('modell.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('modell.pkl','rb'))
print(model.predict([[2, 8, 6]]))
