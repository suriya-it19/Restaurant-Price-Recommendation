import pandas as pd
import os

input = "input"

#data = pd.read_csv(os.path.join(input , "hotel2.csv"))
#data = pd.read_csv(os.path.join(input , "hf.csv"))
data = pd.read_excel(os.path.join(input , "hf.xlsx"))

data.head()


data['year'].unique()


data.replace({
    2015 : 0,
    2016 : 1,
    2017 : 2,
    2018 : 3,
    2019 : 4,
    2020 : 5
    }, inplace=True)


for i in range(3, 25, 3):
    data.iloc[:, i].replace([0,1,2,3,4], 0, inplace=True)
    data.iloc[:, i].replace([5,6,7,8,9,10], 1, inplace=True)


#data.iloc[:, 3:11] =  data.iloc[:, 3:11].replace([0,1,2,3,4], 0)#,inplace=True


#data.iloc[:, 3:11] = data.iloc[:, 3:11].replace([5,6,7,8,9,10], 1)


data


data.columns


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, ensemble
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X = data[['year', 'month', 'week']]
y = data['pure total']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=55)

params = {'n_estimators': 1500,
          'max_depth': 4,
          'min_samples_split': 6,
          'learning_rate': 0.01,
          'loss': 'ls'}

reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_train, y_train)

mse = mean_squared_error(y_test, reg.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

print(r2_score(y_test, reg.predict(X_test)))
reg.predict([[1,12,4]])


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def dish_price(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42)
    
    params = {'n_estimators': 2000,
          'max_depth': 4,
          'min_samples_split': 6,
          'learning_rate': 0.01,
          'loss': 'ls'}

    reg = ensemble.GradientBoostingRegressor(**params)
    reg.fit(X_train, y_train)
    acc = r2_score(y_test, reg.predict(X_test))
    return {f'model_{i}':reg,'acc':acc}

price = ['chicken_curry_price ', 'mutton_chops_price', 'prawn_fry_price', 'chicken_nuggets_price',
       'chettinad_mutton_price',
       'dhal_makni_price',
       'veg_manchurian_price',
       'paneer_tikka_price']
#price = ['chicken_curry', 'mutton_chops', 'prawn_fry', 'chicken_nuggets', 'chettinad_mutton', 'dhal_makni', 'veg_manchurian', 'paneer_tikka']
final_price = []

for i in price:
    X = data[['year','month','week']]
    y = data[[i]]
    final_price.append(dish_price(X, y))
    
final_price


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def solds(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42)
    
    params = {'n_estimators': 2000,
          'max_depth': 4,
          'min_samples_split': 6,
          'learning_rate': 0.01,
          'loss': 'ls'}

    reg = ensemble.GradientBoostingRegressor(**params)
    reg.fit(X_train, y_train)
    acc = r2_score(y_test, reg.predict(X_test))
    return {f'model_{i}':reg,'acc':acc}

sold = ['chicken_curry', 'mutton_chops', 'prawn_fry', 'chicken_nuggets', 'chettinad_mutton', 'dhal_makni', 'veg_manchurian', 'paneer_tikka']
final_sold = []

for i in sold:
    X = data[['year','month','week']]
    y = data[[f'{i}_plates_sold']]
    final_sold.append(solds(X, y))
    
final_sold


from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn import metrics

def rating(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=13)

    clf = AdaBoostClassifier(n_estimators=1500, random_state=13)
    clf.fit(X_train, y_train)
    acc = metrics.accuracy_score(y_test, clf.predict(X_test).reshape(-1, 1))
    return {f'model_{i}':clf,'acc':acc}

ratings = ['chicken_curry', 'mutton_chops', 'prawn_fry', 'chicken_nuggets', 'chettinad_mutton', 'dhal_makni', 'veg_manchurian', 'paneer_tikka']
final_rating = []

for i in ratings:
    X = data[['year','month','week']]
    y = data[[f'{i}']]
    final_rating.append(rating(X, y))
    
final_rating


# mlp for multi-label classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

# get the dataset
def get_dataset():
    #X, y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=3, n_labels=2, random_state=1)
    X = data[['year','month','week']].values#'pure total','dishes sold ratings total','listed total'
    y = data[['chicken curry','mutton chops','prawn fry','chicken nuggets','chettinad mutton','dhal makni','veg manchurian','panner tikka']].values
    return X, y

# get the model
def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(30, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(30, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(20, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y):
    results = list()
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    # define evaluation procedure
    cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=1)
    # enumerate folds
    for train_ix, test_ix in cv.split(X):
        # prepare data
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        # define model
        model = get_model(n_inputs, n_outputs)
        # fit model
        model.fit(X_train, y_train, epochs=500)#, verbose=0
        # make a prediction on the test set
        yhat = model.predict(X_test)
        # round probabilities to class labels
        yhat = yhat.round()
        # calculate accuracy
        acc = accuracy_score(y_test, yhat)
        # store result
        print('>get_ipython().run_line_magic(".3f'", " % acc)")
        results.append(acc)
    return results, model

# load dataset
X, y = get_dataset()
# evaluate model
results, model = evaluate_model(X, y)
# summarize performance
print('Accuracy: get_ipython().run_line_magic(".3f", " (%.3f)' % (mean(results), std(results)))")


# Regression Example With Boston Dataset: Baseline
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

X = data[['year','month','week']].values#'pure total','dishes sold ratings total','listed total'
y = data[['pure total']].values
n_inputs, n_outputs = X.shape[1], y.shape[1]

# define base model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(32, input_dim=n_inputs, kernel_initializer='normal', activation='relu'))
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(n_outputs, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# evaluate model
estimator = KerasRegressor(build_fn=baseline_model, epochs=500, batch_size=5, verbose=0)
kfold = KFold(n_splits=2)
results = cross_val_score(estimator, X, y, cv=kfold)
print("Baseline: get_ipython().run_line_magic(".2f", " (%.2f) MSE\" % (results.mean(), results.std()))")


# mlp for multi-label classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

#X, y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=3, n_labels=2, random_state=1)
X = data[['year','month','week']].values#'pure total','dishes sold ratings total','listed total'
y = data[['chicken curry','mutton chops','prawn fry','chicken nuggets','chettinad mutton','dhal makni','veg manchurian','panner tikka']].values
    
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, shuffle=True)

n_inputs, n_outputs = X.shape[1], y.shape[1]
    
model = Sequential()
model.add(Dense(32, input_dim=n_inputs, kernel_initializer='normal', activation='relu'))
#model.add(Dense(32,  activation='relu'))
model.add(Dense(64,  activation='relu'))
#model.add(Dense(64,  activation='relu'))
#model.add(Dense(128,  activation='relu'))
#model.add(Dense(128,  activation='relu'))
#model.add(Dense(1024,  activation='relu'))
#model.add(Dense(1024,  activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')
    
model.fit(X_train, y_train, epochs=500)
#yhat = model.predict(X_test)
#yhat = yhat.round()
#acc = accuracy_score(y_test, yhat)
#print('>get_ipython().run_line_magic(".3f'", " % acc)")
        
#print('Accuracy: get_ipython().run_line_magic(".3f", " (%.3f)' % (mean(results), std(results)))")


model.predict([[1,1,2]])


import pickle

#
# Create your model here (same as above)
#

# Save to file in the current working directory
pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)

# Load from file
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)
    
# Calculate the accuracy score and predict target values
score = pickle_model.score(Xtest, Ytest)
print("Test score: {0:.2f} get_ipython().run_line_magic("".format(100", " * score))")
Ypredict = pickle_model.predict(Xtest)




































