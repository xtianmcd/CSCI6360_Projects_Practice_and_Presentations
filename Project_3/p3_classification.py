import pandas as pd
import numpy as np
import tensorflow as tf
import random as rn
from keras.initializers import Constant
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import BatchNormalization
from keras.callbacks import CSVLogger, TensorBoard, EarlyStopping
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, ParameterSampler
import os
import time
import csv
"""
__author__ = "Jeremy Shi and Christian McDaniel"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Jeremy Shi and Christian McDaniel"
__email__ = "jeremy.shi@uga.edu, clm121@uga.edu"

This file reads in a dataset and optimizes a densely-connected neural network
on various different hyperparameters using Keras and Sci-kit Learn.

Parameters are sampled randomly from a search grid containing over 100,000
combinations.

Each parameter sampling is used to train a network using 5-fold cross
validation. To avoid any data or meta-data leakage, the data is standardized
separately for each fold. Once the test fold has been separated, the training
folds are standardized, and the testing data is standardized separately using
the training data mean and standard deviation.

The model parameter combinations that yield the five highest scores for
accuracy as well as the five highest F1 Scores are saved to a csv. Additionally,
the performance measures (accuracy, precision, recall, F1 score, true positives,
true negatives, false positives and false negatives) for all parameter
combinations are savedt to a file. Finally, all model training logs are saved
for later examination.

"""

def fix_seeds():
    """
    Fixes the seeds for reproducibility.
    """
    # fix random seed for reproducibility
    os.environ['PYTHONHASHSEED'] = '0'

    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    seed = np.random.seed(42)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(12345)
    tf.set_random_seed(1234)
    return

def prep_data(filepath, debug=False):
    """
    Reads in the data as a CSV and partitions into features vs. target values.
    Also converts the String target values into binary integers (i.e., 0 and 1)

    Parameters:
    -----------------
    filepath: String containing filepath to data on local machine
    debug:    Boolean indicating whether or not to print additional information;
              default=False

    Returns:
    -----------------
    X:         Pandas DataFrame containing training data as floating point nums
    encoded_y: Pandas Series containing target values converted to integers

    """
    df = pd.read_csv("./data/caravan.csv")
    data = df.values
    if debug:
        print("Full DataFrame preview and shape")
        print(data.head())
        print(data.shape)

    X = data[:,:-1].astype(float)
    if debug:
        print("Training data preview and shape")
        print(X.shape)
        print(X.head())

    y = data[:, -1]

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_y = encoder.transform(y)
    if debug:
        print("Target values shape")
        print(y.shape)

    return X, encoded_y

def build_model(n_layers=3, units1=85, units2=45, units3=10, activationh='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', optimizer='adam'):
    """
    Builds the densely-connected neural network using Keras and the provided
    parameters. All parameters given conventionally-used default parameters
    so that the DNN can be trained without explicitly entering the parameters.

    Parameters:
    -----------------
    n_layers: indicates the number of layers
    units1:             denotes the number of hidden units for the first layer
    units2:             denotes the number of hidden units for the second layer,
                        if applicable
    units3:             denotes the number of hidden units for the third through
                        tenth layer, if applicable
    activationh:        activation function used for the hidden layers
    use_bias:           indicates whether or not to use bias for training
    kernel_initializer: distribution used for initializing the kernel
    bias_initializer:   distribution used for initializing bias, if applicable
    optimizer:          optimizer used during back propogation

    Returns:
    -----------------
    model:              the constructed model
    """

    model = Sequential()
    model.add(Dense(units=units1, input_dim=85, activation=activationh, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    if n_layers > 1:
        model.add(Dense(units=units2, activation=activationh, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
    if n_layers > 2:
        model.add(Dense(units=units3, activation=activationh, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
    if n_layers> 3:
        model.add(Dense(units=units3, activation=activationh, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(units=units3, activation=activationh, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
    if n_layers> 5:
        model.add(Dense(units=units3, activation=activationh, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(units=units3, activation=activationh, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(units=units3, activation=activationh, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(units=units3, activation=activationh, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(units=units3, activation=activationh, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid', use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))

    model.compile(optimizer=optimizer,
          loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def check_preds(predictions, actual, debug=False):
    """
    Checks the predictions made by test_model() for correctness.

    Parameters:
    -----------------
    predictions: predicted values returned by test_model()
    actual:      ground truth target values
    debug:       Boolean indicating whether or not to print additional
                 information; default=False

    Returns:
    -----------------
    tp: number of true  positive predictions
    fp: number of false positive predictions
    tn: number of true  negative predictions
    fn: number of false negative predictions
    """

    fp = 0.0
    fn = 0.0
    tp = 0.0
    tn = 0.0
    if debug:
        print('len prediction: {}\nlen actuals: {}'.format(len(predictions), len(actual)))
    for y in range(len(predictions)):
        if predictions[y] == 0:
            if actual[y] == 0: tn += 1.0
            elif actual[y] == 1: fn += 1.0
            else: print("No actuals = 0 or 1 when prediction = 0")
        elif predictions[y] == 1:
            if actual[y] == 1: tp += 1.0
            elif actual[y] == 0: fp += 1.0
            else: print("No actual = 0 or 1 when predictions = 1")
        else: print("No predictions = 0 or 1")

    return tp, tn, fp, fn

def get_stats(tp, tn, fp, fn):
    """
    Computes performace measures using the correctness measures returned by
    check_preds()

    Parameters:
    -----------------
    tp: number of true  positive predictions returned by check_preds()
    fp: number of false positive predictions returned by check_preds()
    tn: number of true  negative predictions returned by check_preds()
    fn: number of false negative predictions returned by check_preds()

    Returns:
    -----------------
    acc:       (tp + tn) / (tp + tn + fp + fn)
    recall:    tp / (tp + fn)
    precision: tp / (tp + fp)
    fscore:    F1 Score, 2 * precision * recall / (precision + recall)
    """
    acc = (tp + tn) / (tp + tn + fp + fn)
    recall = 0.0
    precision = 0.0
    fscore = 0.0

    if tp == 0.0:
        if fp == 0.0 and fn == 0.0:
            recall, precision, fscore = 1.0, 1.0, 1.0
        else:
            recall, precision, fscore = 0.0, 0.0, 0.0
    else:
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        fscore = 2 * precision * recall / (precision + recall)

    return (acc, recall, precision, fscore)

def test_model(X, encoded_y, param_dict={}, model_num=-1, debug=False):
    """
    Performs 5-fold cross validation on the DNN containing the given parameters,
    as part of a pipeline which also standardizes the data after separating the
    test fold for each fold. If no parameters provided, trains the model using
    the default parameters.

    Calls check_preds() and get_stats() to compute the performance measures on
    the predictions and averages them across folds.

    Parameters:
    -----------------
    X: training data returned by prep_data()
    encoded_y:  target values returnd by prep_data()
    param_dict: dictionary of parameter values; default= empty dict, which
                prompts build_model() to use its default parameters for
                constructing the model
    model_num:  used for printing and saving, if parameter combinations are being
                provided iteratively; default=-1, useful in indicating
                no specific model parameters are provided (no iterator being used)
    debug:      Boolean indicating whether or not to print additional
                information; default=False

    Returns:
    -----------------
    mean_acc: mean accuracy across folds
    std_acc: accuracy stdard deviation across folds
    mean_fscore: mean F1 Score across folds
    std_fscore: F1 Score standard deviation across folds
    mean_tp: mean true positive  predictions across folds
    mean_tn: mean true negative  predictions across folds
    mean_fn: mean false positive predictions across folds
    mean_fp: mean false negative predictions across folds
    mean_recall: mean recall across folds
    std_recall: recall standard deviation across folds
    mean_precis: mean precision across folds
    std_precis: precision standard deviation across folds
    """

    print("testing model {}".format(model_num))
    model_acc = []
    model_recall = []
    model_precis = []
    model_fscore = []
    time_elapsed = []
    model_tp = []
    model_tn = []
    model_fp = []
    model_fn = []
    split = 1

    # initiate cross validation
    skf = StratifiedKFold(n_splits=5)
    for train_index, test_index in skf.split(X, encoded_y):
        if param_dict:
            model = build_model(n_layers=param_dict['n_layers'], units1=param_dict['units1'], units2=param_dict['units2'], units3=param_dict['units3'], activationh=param_dict['activationh'], use_bias=param_dict['use_bias'], kernel_initializer=param_dict['kernel_initializer'], bias_initializer=param_dict['bias_initializer'], optimizer=param_dict['optimizer'])
        else: model = build_model()
        if debug:
            print("training fold {}".format(split))
        start = time.time()

        #standardize training and test data
        scaler = StandardScaler()
        scaler.fit(X[train_index])
        X_train = scaler.transform(X[train_index])
        X_test  = scaler.transform(X[test_index])

        X_tr = X[train_index]
        X_te = X[test_index]
        y_tr = encoded_y[train_index]
        y_te = encoded_y[test_index]

        # Separate target values into training and ground truth for testing
        y_train = encoded_y[train_index]
        y_test  = encoded_y[test_index]

        # Keras callbacks
        csv_logger = CSVLogger('training_model_{}.log'.format(model_num), append=True)

        # launch TensorBoard via tensorboard --logdir=/full_path_to_your_logs
        #tb_logs = TensorBoard(log_dir='./logs', histogram_freq=10, batch_size=128, write_graph=True, write_grads=True, write_images=True, embeddings_freq=20, embeddings_layer_names=None, embeddings_metadata=None)

        early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0, mode='auto')
        if debug:
            if param_dict:
                model.fit(X_train, y_train, epochs=1000, batch_size=param_dict['batch_size'], validation_split=0.2, callbacks=[csv_logger, early_stop])#, tb_logs])
            else: model.fit(X_train, y_train, epochs=1000, batch_size=128, validation_split=0.2, callbacks=[csv_logger, early_stop])
        elif param_dict:
            model.fit(X_train, y_train, epochs=1000, batch_size=param_dict['batch_size'], validation_split=0.2, verbose=0, callbacks=[csv_logger, early_stop])#, tb_logs])
        else: model.fit(X_train, y_train, epochs=1000, batch_size=128, validation_split=0.2, verbose=0, callbacks=[csv_logger, early_stop])

        # the below lines were used to test the unstandardized data to check for
        # differences in results
        #model.fit(X_tr, y_tr, epochs=1000, batch_size=param_dict['batch_size'], validation_split=0.2, callbacks=[csv_logger, early_stop])#, tb_logs])
        #model.fit(X_tr, y_tr, epochs=10000, batch_size=128, validation_split=0.2, callbacks=[csv_logger, early_stop])

        if debug:
            print("predicting fold {}".format(split))

        # test the model on the testing data
        pred = model.predict(X_test)

        # convert the predictions using 0.6 as a threshold to bias slightly
        # toward "No" for Purchase since it would be worse to Purchase a bad
        # car than not purchase a potentially good car.
        # It should be noted that I initially tested multiple thresholds (0.1,
        # 0.25, 0.4, 0.5, 0.6, 0.75 and 0.9) and they almost always yielded the
        # same binary predictions / performance
        pred_binary = []
        for prediction in pred:
            if prediction < 0.6:
                pred_binary.append(0)
            else: pred_binary.append(1)
            if debug:
                print(pred_stand)
                print(pred_binary)
        if debug:
            print("getting stats fold {}".format(split))

        # Get performance measures
        tp, tn, fp, fn = check_preds(pred_binary, y_test)
        fold_acc, fold_recall, fold_precis, fold_fscore = get_stats(tp, tn, fp, fn)

        model_acc.append(fold_acc)
        model_recall.append(fold_recall)
        model_precis.append(fold_precis)
        model_fscore.append(fold_fscore)
        model_tp.append(tp)
        model_tn.append(tn)
        model_fp.append(fp)
        model_fn.append(fn)
        end = time.time()
        time_elapsed.append(end - start)
        split += 1
    if debug:
        print("done with cv")
    model_acc = np.array(model_acc)
    model_recall = np.array(model_recall)
    model_precis = np.array(model_precis)
    model_fscore = np.array(model_fscore)
    time_elapsed = np.array(time_elapsed)
    model_tp = np.array(model_tp)
    model_tn = np.array(model_tn)
    model_fp = np.array(model_fp)
    model_fn = np.array(model_fn)
    if debug:
        print("writing to file")
    if model_num == -10:        # used in best_model.py to indicate best model
        f = open('best_param_results.txt', 'a')
    else:
        f = open('param_results.txt', 'a')
    f.write("Params fold {}: {}\n".format(model_num, param_dict))
    f.write("Performance:\n\tAccuracy: {}% +/-{}\n\tRecall: {} +/-{}\n\tPrecision: {} +/-{}\n\tF1Score: {} +/-{}".format(np.mean(model_acc)*100, np.std(model_acc), np.mean(model_recall), np.std(model_recall), np.mean(model_precis), np.std(model_precis), np.mean(model_fscore), np.std(model_fscore)))
    f.write("Results: tp:{}, tn:{}, fp:{}, fn:{}".format(np.mean(model_tp), np.mean(model_tn), np.mean(model_fp), np.mean(model_fn)))
    f.write("Time Elapsed: {0:.2f} average per run for total of {0:.2f}".format(np.mean(time_elapsed), np.sum(time_elapsed)))
    f.write('\n\n')
    f.close()

    print("Model Report")
    print("Params fold {}: {}".format(model_num, param_dict))
    print("\nPerformance:\n\tAccuracy: {}% +/-{}\n\tRecall: {} +/-{}\n\tPrecision: {} +/-{}\n\tF1Score: {} +/-{}".format(np.mean(model_acc)*100, np.std(model_acc), np.mean(model_recall), np.std(model_recall), np.mean(model_precis), np.std(model_precis), np.mean(model_fscore), np.std(model_fscore)))
    print("Results: tp:{}, tn:{}, fp:{}, fn:{}".format(np.mean(model_tp), np.mean(model_tn), np.mean(model_fp), np.mean(model_fn)))
    print("\nTime Elapsed: {0:.2f} average per run for total of {0:.2f}".format(np.mean(time_elapsed), np.sum(time_elapsed)))

    mean_acc =    np.mean(model_acc)
    std_acc  =    np.std(model_acc)
    mean_fscore = np.mean(model_fscore)
    std_fscore =  np.std(model_fscore)
    mean_tp =     np.mean(model_tp)
    mean_tn =     np.mean(model_tn)
    mean_fn =     np.mean(model_fn)
    mean_fp =     np.mean(model_fp)
    mean_recall = np.mean(model_recall)
    std_recall =  np.std(model_recall)
    mean_precis = np.mean(model_precis)
    std_precis =  np.std(model_precis)
    return mean_acc, std_acc, mean_fscore, std_fscore, mean_tp, mean_tn, mean_fn, mean_fp, mean_recall, std_recall, mean_precis, std_precis

def write_scores(acc_dict, fscore_dict):
    """
    Writes the parameters, model number and accuracy/F1 Score value for the
    provided model.

    Parameters:
    -----------------
    acc_dict:    dictionary containing the model number, testing accuracy and
                 parameters for the desired model
    fscore_dict: dictionary containing the model number, testing F1 Score and
                 parameters for the desired model
    """

    with open('best_scores.csv', 'w') as best_scores:
        writer = csv.writer(best_scores)
        for key, value in acc_dict.items():
            writer.writerow([key, value])
        for key, value in fscore_dict.items():
            writer.writerow([key, value])
    return


if __name__ == '__main__':
    fix_seeds()                 # fix the seeds for reproducibility
    X, encoded_y = prep_data()  # prepare the data for pipeline

    # Construct the parameter grid
    n_layers = [1,2,3,5,10]
    batch_size = [50, 128, 512, 1280, X.shape[0]]
    activation = ['softmax', 'tanh', 'sigmoid', 'relu', 'linear']
    use_bias = [True, False]
    constant = Constant(value=0.1)
    initializer = ['zeros', 'ones', constant, 'lecun_normal', 'glorot_uniform']
    units = [85, 45, 10]
    sgd_001 = SGD(lr=0.001)
    sgd_01 = SGD(lr=0.01)
    sgd_1 = SGD(lr=0.1)
    sgd1_ = SGD(lr=1)
    optimizer = [sgd_001, sgd_01, sgd_1, sgd1_, 'adam', 'nadam']

    param_grid = dict(n_layers=n_layers, units1=units, units2=units, units3=units, activationh=activation, use_bias=use_bias, kernel_initializer=initializer, bias_initializer=initializer, batch_size=batch_size, optimizer=optimizer)

    stats={}
    best_acc = {}
    best_fscore = {}

    #initialize accuracy/ F1Score dictionary
    for rank in range(1,6):
        best_acc['rank{}_model_num'.format(rank)] = 0
        best_acc['rank{}_acc'.format(rank)] = 0
        best_acc['rank{}_params'.format(rank)] = {}
        best_fscore['rank{}_model_num'.format(rank)] = 0
        best_fscore['rank{}_fscore'.format(rank)] = 0
        best_fscore['rank{}_params'.format(rank)] = {}
    write_scores(best_acc, best_fscore)

    # test model using default parameters
    acc0mean, acc0std, fscore0mean, fscore0std, tp0, tn0, fp0, fn0, recall0mean, recall0std, precis0mean, precis0std = test_model(X, encoded_y)

    # generates random combination of parameters from the parameter grid
    # number of combinations initially set to max possible combinations
    # Note that training a model for each combinations would make this program
    # pretty much run indefinitely unless a massive distributed system is used.
    param_list = list(ParameterSampler(param_grid, n_iter=112500))

    # test the models that result from the combinations of parameters
    # Store the best accuracies and F1 Scores found
    for d in range(len(param_list)):
        acc_mean, acc_std, fscore_mean, fscore_std, tp, tn, fp, fn, recall_mean, recall_std, precis_mean, precis_std = test_model(X, encoded_y, param_list[d], model_num=d)
        tie_acc = 1
        tie_fscore = 1

        if acc_mean > best_acc['rank1_acc']:
            best_acc['rank1_model_num'] = d
            best_acc['rank1_acc'] = acc_mean
            best_acc['rank1_params'] = param_list[d]
            write_scores(best_acc, best_fscore)
        elif acc_mean > best_acc['rank2_acc']:
            best_acc['rank2_model_num'] = d
            best_acc['rank2_acc'] = acc_mean
            best_acc['rank2_params'] = param_list[d]
            write_scores(best_acc, best_fscore)
        elif acc_mean > best_acc['rank3_acc']:
            best_acc['rank3_model_num'] = d
            best_acc['rank3_acc'] = acc_mean
            best_acc['rank3_params'] = param_list[d]
            write_scores(best_acc, best_fscore)
        elif acc_mean > best_acc['rank4_acc']:
            best_acc['rank4_model_num'] = d
            best_acc['rank4_acc'] = acc_mean
            best_acc['rank4_params'] = param_list[d]
            write_scores(best_acc, best_fscore)
        elif acc_mean > best_acc['rank5_acc']:
            best_acc['rank5_model_num'] = d
            best_acc['rank5_acc'] = acc_mean
            best_acc['rank5_params'] = param_list[d]
            write_scores(best_acc, best_fscore)
        elif acc_mean == best_acc['rank5_acc']:
            best_acc['rank5_{}_model_num'.format(tie_acc)] = d
            best_acc['rank5_{}_acc'.format(tie_acc)] = acc_mean
            best_acc['rank5_{}_params'.format(tie_acc)] = param_list[d]
            write_scores(best_acc, best_fscore)
            tie_acc += 1


        if fscore_mean > best_fscore['rank1_fscore']:
            best_fscore['rank1_model_num'] = d
            best_fscore['rank1_fscore'] = fscore_mean
            best_fscore['rank1_params'] = param_list[d]
            write_scores(best_acc, best_fscore)
        elif fscore_mean > best_fscore['rank2_fscore']:
            best_fscore['rank2_model_num'] = d
            best_fscore['rank2_fscore'] = fscore_mean
            best_fscore['rank2_params'] = param_list[d]
            write_scores(best_acc, best_fscore)
        elif fscore_mean > best_fscore['rank3_fscore']:
            best_fscore['rank3_model_num'] = d
            best_fscore['rank3_fscore'] = fscore_mean
            best_fscore['rank3_params'] = param_list[d]
            write_scores(best_acc, best_fscore)
        elif fscore_mean > best_fscore['rank4_fscore']:
            best_fscore['rank4_model_num'] = d
            best_fscore['rank4_fscore'] = fscore_mean
            best_fscore['rank4_params'] = param_list[d]
            write_scores(best_acc, best_fscore)
        elif fscore_mean > best_fscore['rank5_fscore']:
            best_fscore['rank5_model_num'] = d
            best_fscore['rank5_fscore'] = fscore_mean
            best_fscore['rank5_params'] = param_list[d]
            write_scores(best_acc, best_fscore)
        elif fscore_mean == best_fscore['rank5_fscore']:
            best_fscore['rank5_{}_model_num'.format(tie_fscore)] = d
            best_fscore['rank5_{}_fscore'.format(tie_fscore)] = fscore_mean
            best_fscore['rank5_{}_params'.format(tie_fscore)] = param_list[d]
            write_scores(best_acc, best_fscore)
            tie_fscore += 1
