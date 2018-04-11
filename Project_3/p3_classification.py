import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
# from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
import os
import tensorflow as tf
import random as rn

# fix random seed for reproducibility

os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
seed = np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
rn.seed(12345)

tf.set_random_seed(1234)

df = pd.read_csv("./data/caravan.csv")
data = df.values
#print(data.head())
# print(data.shape)

X = data[:,:-1].astype(float)

# print(X.shape)
# print(X.head())

y = data[:, -1]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
# print(y.shape)



def build_model_simple():
    model = Sequential()
    # for layer in range(layers):
    model.add(Dense(85, input_dim=85, activation='relu', kernel_initializer='normal'))
    # if layers >= 2:
    #     model.add(Dense(85, input_dim=85, activation='relu', kernel_initializer='normal'))
    model.add(Dense(1, kernel_initializer = 'normal', activation='sigmoid'))

    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model

def build_model_deeper():
    model = Sequential()
    # for layer in range(layers):
    model.add(Dense(45, input_dim=85, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.4))
    model.add(Dense(25, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer = 'normal', activation='sigmoid'))

    model.compile(optimizer='adam',
          loss='binary_crossentropy',
          metrics=['accuracy'])
    return model

estimator_simple = KerasClassifier(build_fn=build_model_simple)
estimator_deeper = KerasClassifier(build_fn=build_model_deeper)

parameters = {'batch_size' : [10, 20, 40, 80, 160, 320, 640, 1280, 2560, X.shape[0]], 'epochs' : [1, 5, 10, 25, 50, 75, 100, 150, 200]}
#
# grid = GridSearchCV(KerasRegressor(build_fn=build_model), parameters, cv = 2, return_train_score = True)
# for estimator in [estimator_simple, estimator_deeper]:
grid = GridSearchCV(estimator=estimator_simple, param_grid=parameters, return_train_score = True)
grid_result = grid.fit(X, encoded_y)

    # evaluate model with standardized dataset
    # estimator = KerasClassifier(build_fn=build_model, epochs=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(grid, X, encoded_y, cv=kfold)
    # print("Results: {:.2f}% ({:.2f}%)".format(results.mean()*100, results.std()*100))
print("Best: {} using {}".format(grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("{} ({}) with: {}".format(mean, stdev, param))
