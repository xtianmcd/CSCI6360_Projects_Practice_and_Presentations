import pandas as pd
import numpy as np
import tensorflow as tf
import random as rn
from sklearn.svm import SVC, LinearSVC, SVR 
from keras.initializers import Constant
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import BatchNormalization
from keras.callbacks import CSVLogger, TensorBoard, EarlyStopping
from sklearn.preprocessing import LabelEncoder, StandardScaler, LabelBinarizer
from sklearn.model_selection import StratifiedKFold, ParameterSampler
import os
import time
import csv
import sys

"""
__author__ = "Jeremy Shi and Christian McDaniel"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Jeremy Shi and Christian McDaniel"
__email__ = "jeremy.shi@uga.edu, clm121@uga.edu"

This file can accommodate datasets with both binary or multi-class target values
and can run a myriad of classification as well as one regression tests on the
data. The models supported include a densely-connected neural network, SVM-C,
linear SVM and SVM regressor. Each model is optimized regarding on various
different hyperparameters using Sci-kit Learn.

Parameters are sampled randomly from a search grid.

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

To run the file from command line, navigate to the folder containing this file
and enter the executable: "python " followed by the following arguments:

    arg 0: name of script (e.g., fp_classification.py)
    arg 1: filepath to dataset (e.g., './winequality-white.csv')
    arg 2: model type, either 'nn', 'svm', 'linearsvm' or 'svr'
    arg 3: string indicating if the target variable is binary;
           if not binary, provide empty string (i.e., ''),
           else enter any string (e.g., True, 'binary', 'yes')


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

def prep_data(filepath, binary, modeltype, sep=',', debug=False):
    """
    Reads in the data as a CSV and partitions into features vs. target values.
    Also converts the String target values into binary integers (i.e., 0 and 1)

    Parameters:
    -----------------
    filepath:  String containing filepath to data on local machine
    binary:    Boolean indicating if target value is binary
    modeltype: model type given as argument at commant line
    sep:       Character separating the columns in the datafile, used by Pandas
               default=','
    debug:     Boolean indicating whether or not to print additional information;
               default=False

    Returns:
    -----------------
    X:              Pandas DataFrame containing training data as floating point nums
    encoded_y:      Pandas Series containing target values converted to integers
    y:              original target values from raw data
    k:              number of classes in target variable
    num_predictors: number of features in dataset, used by neural network for
                    input dimension

    """
    df = pd.read_csv(filepath, sep=sep)
    data = df.values
    if debug:
        print("Full DataFrame preview and shape")
        print(df.head())
        print(data.shape)

    X = data[:,:-1].astype(float)
    num_predictors = X.shape[1]

    if debug:
        print("Training data preview and shape")
        print(X.shape)

    y = data[:, -1]
    k = len(np.unique(y))


    if model=='nn' and not binary:      #use one-hot labels
        lb = LabelBinarizer()
        lb.fit(y)
        encoded_y = lb.transform(y)
        k=encoded_y.shape[1]
    else:                                # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(y)
        encoded_y = encoder.transform(y)
        if binary: k=1

    if debug:
        print("Target values shape: {}".format(y.shape))
        print("Encoded target values shape: {}".format(encoded_y.shape))

    return X, encoded_y, y, k, num_predictors

def build_model(num_features, num_classes, binary, n_layers=3, units1=85, units2=45, units3=10, activationh='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', optimizer='adam'):
    """
    Builds the densely-connected neural network using Keras and the provided
    parameters. All parameters given conventionally-used default parameters
    so that the DNN can be trained without explicitly entering the parameters.

    Parameters:
    -----------------
    num_features:       number of features in data, returnd by prep_data()
    num_classes:        num classes in target variable, returned by prep_data()
    binary:             Boolean indicating whether target variable is binary
    n_layers:           indicates the number of layers
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
    model.add(Dense(units=units1, input_dim=num_features, activation=activationh, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
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

    model.add(Dense(num_classes, activation='sigmoid', use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))

    if binary:
        model.compile(optimizer=optimizer,
          loss='binary_crossentropy',
                  metrics=['accuracy'])
    else:
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model

def binarize_preds(predictions, threshold=0.5, debug=False):
    """
    Converts real-value predictions between 0 and 1 to integer class values 0
    and 1 according to a threshold.

    Parameters:
    -----------------
    predictions: predicted values
    threshold:   threshold for converting prediction to 0 or 1
    debug:       Boolean indicating whether or not to print additional information;
                 default=False

    Returns:
    -----------------
    pred_binary: binarized predictions
    """
    pred_binary = []
    for prediction in predictions:
        if prediction < threshold:
            pred_binary.append(0)
        else: pred_binary.append(1)
        if debug:
            print(predictions)
            print(pred_binary)
    return pred_binary

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
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0

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

    return tp, fp, tn, fn

#This method was buggy... needs fixing
# def multiclass_accuracy(predictions, actual, k, debug=False):
#     class_acc = {}
#
#     ls = 1/k # laplace smoothing
#     total_corr=0
#     total_ex = actual.shape[0]
#     for c in range(k):
#         correct = 0
#         class_size = 0
#         correct = sum([correct+1 for i in range(actual.shape[0]) if np.argmax(predictions[i]) == c and np.argmax(actual[i]) == c])
#         class_size = sum([class_size+1 for i in range(actual.shape[0]) if np.argmax(actual[i]) == c])
#         accuracy = 100 * ((correct+ls) / (class_size+ls*k))
#         class_acc[str(c)] = accuracy
#         total_corr+=correct
#         if debug: print("Class {}: Predicted {} out of {} correctly for an Accuracy of {}%".format(c, correct, class_size, accuracy))
#     overall_acc = total_corr/total_ex
#     return class_acc, overall_acc

def multiclass_pred_check(predictions, actual, k, regression, tol=0.5, debug=False):
    """
    Calculates the tp, fp, tn and fn statistics from each class. If
    classficiation, takes the index of the maximum value of the one-hot
    prediction as the predicted class. If regression, computes statistics based
    on the provided tolerance value (T). T=0.5 means the nearest integer is the
    predicted class, while T=1.0 means the resulting integer from both the
    floor function or the ceiling function of the prediction are considered
    (more lenient). Stores class-wise statistics as well as sums each together
    for full fold-wise statistics across all classes.

    Parameters:
    -----------------
    predictons: predicted values
    actual:     ground truth values
    k:          num classes
    debug:      Boolean indicating whether or not to print additional
                information; default=False

    Returns:
    -----------------
    tp: number of true  positive predictions
    fp: number of false positive predictions
    tn: number of true  negative predictions
    fn: number of false negative predictions
    ctp: dictionary with number of true  positive predictions for each class
    cfp: dictionary with number of false positive predictions for each class
    ctn: dictionary with number of true  negative predictions for each class
    cfn: dictionary with number of false negative predictions for each class

    """
    tp=0
    fp=0
    tn=0
    fn=0
    ctp = {}
    cfp={}
    ctn={}
    cfn={}
    for c in range(k):
        tpc=0
        fpc=0
        tnc=0
        fnc=0

        if not regression:
            tpc = sum([tpc + 1 for i in range(actual.shape[0]) if np.argmax(predictions[i]) == c and np.argmax(actual[i]) == c])
            fpc = sum([fpc + 1 for i in range(actual.shape[0]) if np.argmax(predictions[i]) == c and np.argmax(actual[i]) != c])
            tnc = sum([tnc + 1 for i in range(actual.shape[0]) if np.argmax(predictions[i]) != c and np.argmax(actual[i]) != c])
            fnc = sum([fnc + 1 for i in range(actual.shape[0]) if np.argmax(predictions[i]) != c and np.argmax(actual[i]) == c])
        else:
            if tol==0.5:
                tpc = sum([tpc + 1 for i in range(len(actual)) if predictions[i] == c and actual[i] == c])
                fpc = sum([fpc + 1 for i in range(len(actual)) if predictions[i] == c and actual[i] != c])
                tnc = sum([tnc + 1 for i in range(len(actual)) if predictions[i] != c and actual[i] != c])
                fnc = sum([fnc + 1 for i in range(len(actual)) if predictions[i] != c and actual[i] == c])
            elif tol==1.0:
                tpc = sum([tpc + 1 for i in range(len(actual)) if any(prediction == c for prediction in predictions[i]) and np.argmax(actual[i]) == c])
                fpc = sum([fpc + 1 for i in range(len(actual)) if any(prediction == c for prediction in predictions[i]) and np.argmax(actual[i]) != c])
                tnc = sum([tnc + 1 for i in range(len(actual)) if any(prediction != c for prediction in predictions[i]) and np.argmax(actual[i]) != c])
                fnc = sum([fnc + 1 for i in range(len(actual)) if any(prediction != c for prediction in predictions[i]) and np.argmax(actual[i]) == c])
        if debug: print("class {} tp: {}; fp: {}; tn: {}; fn: {}".format(c, tpc, fpc, tnc, fnc))
        tp+=tpc
        fp+=fpc
        tn+=tnc
        fn+=fnc
        ctp[str(c)]=tpc
        cfp[str(c)]=fpc
        ctn[str(c)]=tnc
        cfn[str(c)]=fnc
    if debug: print("fold tp: {}, fp: {}, tn: {}, fn: {}".format(tp, fp, tn, fn))
    return tp, fp, tn, fn, ctp, cfp, ctn, cfn

def get_stats(tp, fp, tn, fn):
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

    return acc, recall, precision, fscore

def calc_error(predictions, actual, debug=False):
    """
    Computes the mean absolute error for the regression prediction values

    Parameters:
    -----------------
    predictions: predicted values
    actual:      actual values
    debug:       Boolean indicating whether or not to print additional
                 information; default=False

    Returns:
    -----------------
    mae: computed mean absolute error

    """
    if debug: print("sample predictions: {}".format([predictions[i] for i in range(10)]))
    # pred = [int(round(prediction)) for prediction in predictions]
    # if debug: print("sample predictions after casting: {}".format([pred[i] for i in range(10)]))
    if debug: print("sample target labels: {}".format([actual[i] for i in range(10)]))
    mae = sum([abs(predictions[i]-actual[i]) for i in range(len(actual))]) / len(actual)
    return mae

def regressor_accuracy(predictions, tolerance=0.5, debug=False):
    if debug: print("sample predictions: {}".format([predictions[i] for i in range(10)]))
    if tolerance == 0.5:
        pred = [int(round(prediction)) for prediction in predictions]
    if tolerance == 1.0:
        pred = [[math.floor(prediction), math.ceil(prediction)] for prediction in predictions]
    if debug: print("sample predictions after casting: {}".format([pred[i] for i in range(10)]))
    return pred

def test_model(X, encoded_y, y, feat_num, class_num, binary, modeltype, verbose=False, param_dict={}, model_num=-1, tolerance=0.5, debug=False):
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
    y:          original target values
    feat_num:   number of predictors
    binary:     Boolean indicating if target variable is binary
                default=True
    param_dict: dictionary of parameter values; default= empty dict, which
                prompts build_model() to use its default parameters for
                constructing the model
    model_num:  used for printing and saving, if parameter combinations are being
                provided iteratively; default=-1, useful in indicating
                no specific model parameters are provided (no iterator being used)
    tolerance:  tolerance used for regression accuracy calculation
                default=0.5
    verbose:    Boolean indicating whether or not to print statistics to console
                default=False
    debug:      Boolean indicating whether or not to print additional
                information; default=False

    Returns:
    -----------------
    model_acc[0]:        mean model accuracy
    model_fscore[0]:     mean model F1 Score
    model_classwise_acc: dictionary of classwise accuracies
                         (if multiclass target variable)
    """

    print("testing model {}".format(model_num))
    acc = []
    recall = []
    precis = []
    fscore = []

    tp = []
    fp = []
    tn = []
    fn = []

    r2 = []
    err=[]

    time_elapsed = []
    mc_overall_acc = []
    mc_classwise_acc = {}
    mc_classwise_precis={}
    mc_classwise_recall={}
    mc_classwise_fscore={}
    for k in range(n_classes):
        mc_classwise_acc[str(k)]=[]
        mc_classwise_recall[str(k)]=[]
        mc_classwise_precis[str(k)]=[]
        mc_classwise_fscore[str(k)]=[]

    tol=tolerance
    if not binary:
        model_class_acc = {}
        for k in range(class_num):
            model_class_acc[str(k)]=[]
    split = 1

    # initiate cross validation
    skf = StratifiedKFold(n_splits=5)
    for train_index, test_index in skf.split(X, y):
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
        regress = False

        if modeltype == 'nn':
            if param_dict:
                model = build_model(feat_num, class_num, binary=binary, n_layers=param_dict['n_layers'], units1=param_dict['units1'], units2=param_dict['units2'], units3=param_dict['units3'], activationh=param_dict['activationh'], use_bias=param_dict['use_bias'], kernel_initializer=param_dict['kernel_initializer'], bias_initializer=param_dict['bias_initializer'], optimizer=param_dict['optimizer'])
            else: model = build_model(feat_num, class_num, binary=binary)
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

        elif modeltype == 'svm':
            if param_dict:
                model = SVC(param_dict, probability=True, random_state=None)
            else: model = SVC(probability=True)
            model.fit(X_train, y_train)

            score = model.score(X_test, y_test)
            acc.append(score)

        elif modeltype == 'linearsvm':
            if param_dict:
                model = LinearSVC(loss=param_dict['loss'], C=param_dict['C'], multi_class=param_dict['mult_class'])
            else: model = LinearSVC()

            model.fit(X_train, y_train)

            score = model.score(X_test, y_test)
            acc.append(score)

        elif modeltype == 'svr':
            if param_dict:
                model = SVR(epsilon=param_dict['ep'], C=param_dict['C'], kernel=param_dict['kernel'], degree=param_dict['degree'], gamma=param_dict['gamma'])
            else: model = SVR()

            model.fit(X_train, y_train)

            score = model.score(X_test, y_test)
            r2.append(score)
            regress=True

        else: print("Model type not supported")

        if debug: print("predicting fold {}".format(split))

        # test the model on the testing data
        pred = model.predict(X_test)
        if debug:
            print("Sample prediction: {}".format(pred[0]))
            print("First 5 predictions followed by first 5 target labels:")
            print(list(np.argmax(pred[i]) for i in range(5)))
            print(list(np.argmax(y_test[i]) for i in range(5)))
            print("{} classes".format(class_num))
            if y_test.shape[0] != pred.shape[0]: print("Predictions and labels not the same length!")

        if modeltype == 'svr':
            fold_err = calc_error(pred, y_test)

            pred = regressor_accuracy(pred, tolerance=tol)
            err.append(fold_err)

        if binary:
            pred_binary = binarize_preds(pred, threshold=0.5)
            #get performance measures
            fold_tp, fold_fp, fold_tn, fold_fn = check_preds(pred_binary, y_test)
            fold_acc, fold_recall, fold_precis, fold_fscore = get_stats(fold_tp, fold_fp, fold_tn, fold_fn)

        else:
            if debug: print("y_test shape: {}".format(y_test.shape))
            fold_tp, fold_fp, fold_tn, fold_fn, fold_class_tp, fold_class_fp, fold_class_tn, fold_class_fn = multiclass_pred_check(pred, y_test, n_classes, regression=regress, tol=tol)
            fold_acc, fold_recall, fold_precis, fold_fscore = get_stats(fold_tp, fold_fp, fold_tn, fold_fn)

            for k in range(n_classes):
                fold_cw_acc, fold_cw_recall, fold_cw_precis, fold_cw_fscore = get_stats(fold_class_tp[str(k)], fold_class_fp[str(k)], fold_class_tn[str(k)], fold_class_fn[str(k)])
                mc_classwise_acc[str(k)].append(fold_cw_acc)
                mc_classwise_recall[str(k)].append(fold_cw_recall)
                mc_classwise_precis[str(k)].append(fold_cw_precis)
                mc_classwise_fscore[str(k)].append(fold_cw_fscore)

        if modeltype != 'svm' and modeltype!= 'linearsvm': acc.append(fold_acc)
        recall.append(fold_recall)
        precis.append(fold_precis)
        fscore.append(fold_fscore)
        tp.append(fold_tp)
        fp.append(fold_fp)
        tn.append(fold_tn)
        fn.append(fold_fn)

        if debug: print(fold_tp, fold_fp, fold_tn, fold_fn)

        if debug: print(fold_acc, fold_recall, fold_precis, fold_fscore)

        if debug: print("getting stats fold {}".format(split))

        end = time.time()
        time_elapsed.append(end - start)
        split += 1
    if debug: print("done with cv")

    model_acc = [np.mean(np.array(acc)), np.std(np.array(acc))]
    model_recall = [np.mean(np.array(recall)), np.std(np.array(recall))]
    model_precis = [np.mean(np.array(precis)), np.std(np.array(precis))]
    model_fscore = [np.mean(np.array(fscore)), np.std(np.array(fscore))]

    model_tp = np.mean(np.array(tp))
    model_fp = np.mean(np.array(fp))
    model_tn = np.mean(np.array(tn))
    model_fn = np.mean(np.array(fn))

    if modeltype=='svr':
        model_r2 = [np.mean(np.array(r2)), np.std(np.array(r2))]
        model_err = [np.mean(np.array(err)), np.std(np.array(err))]

    if not binary:
        model_classwise_acc = {}
        model_classwise_precis={}

        for k in range(n_classes):
            model_classwise_acc[str(k)] = [np.mean(np.array(mc_classwise_acc[str(k)])), np.std(np.array(mc_classwise_acc[str(k)]))]
            model_classwise_precis[str(k)] = [np.mean(np.array(mc_classwise_precis[str(k)])), np.std(np.array(mc_classwise_precis[str(k)]))]


    if debug: print("writing to file")
    if model_num == -10:        # used in best_model.py to indicate best model
        f = open('best_param_results.txt', 'a')
    else:
        f = open('param_results.txt', 'a')
    f.write("Params for model {0:.4f}: {0:.4f}\n".format(model_num, param_dict))
    if binary: f.write("Performance:\n\tAccuracy: {0:.4f}% +/-{0:.4f}\n\tRecall: {0:.4f} +/-{0:.4f}\n\tPrecision: {0:.4f} +/-{0:.4f}\n\tF1Score: {0:.4f} +/-{0:.4f}\n".format(model_acc[0]*100, model_acc[1], model_recall[0], model_recall[1], model_precis[0], model_precis[1], model_fscore[0], model_fscore[1]))
    else:
        f.write("Overall Performance:\n\tAccuracy: {0:.4f}% +/-{0:.4f}\n\tRecall: {0:.4f} +/-{0:.4f}\n\tPrecision: {0:.4f} +/-{0:.4f}\n\tF1Score: {0:.4f} +/-{0:.4f}\n".format(model_acc[0]*100, model_acc[1], model_recall[0], model_recall[1], model_precis[0], model_precis[1], model_fscore[0], model_fscore[1]))
        if modeltype=='svr': f.write("R2: {0:.4f} +/-{0:.4f}\n\tMAE: {0:.4f} +/-{0:.4f}\n".format(model_r2[0], model_r2[1], model_err[0], model_err[1]))
        f.write("Class-specific Accuracies:\n\t")
        for k in range(n_classes):
            f.write("Class {}: {}% +/-{}; ".format(k, model_classwise_acc[str(k)][0], model_classwise_acc[str(k)][1]))
        f.write('\n')
        f.write("Class-specific Precisions:\n\t")
        for k in range(n_classes):
            f.write("Class {}: {}% +/-{}; ".format(k, model_classwise_precis[str(k)][0], model_classwise_precis[str(k)][1]))
        f.write('\n')
    f.write("Results: tp:{}, tn:{}, fp:{}, fn:{}\n".format(model_tp, model_tn, model_fp, model_fn))
    f.write("Time Elapsed: {0:.4f} average per run for total of {0:.4f}".format(np.mean(np.array(time_elapsed)), np.sum(np.array(time_elapsed))))
    f.write('\n\n')
    f.close()

    if verbose:
        print("Params for model {0:.4f}: {0:.4f}\n".format(model_num, param_dict))
        if binary: print("Performance:\n\tAccuracy: {0:.4f}% +/-{0:.4f}\n\tRecall: {0:.4f} +/-{0:.4f}\n\tPrecision: {0:.4f} +/-{0:.4f}\n\tF1Score: {0:.4f} +/-{0:.4f}\n".format(model_acc[0]*100, model_acc[1], model_recall[0], model_recall[1], model_precis[0], model_precis[1], model_fscore[0], model_fscore[1]))
        else:
            print("Overall Performance:\n\tAccuracy: {0:.4f}% +/-{0:.4f}\n\tRecall: {0:.4f} +/-{0:.4f}\n\tPrecision: {0:.4f} +/-{0:.4f}\n\tF1Score: {0:.4f} +/-{0:.4f}\n".format(model_acc[0]*100, model_acc[1], model_recall[0], model_recall[1], model_precis[0], model_precis[1], model_fscore[0], model_fscore[1]))
            if modeltype=='svr': print("R2: {0:.4f} +/-{0:.4f}\n\tMAE: {0:.4f} +/-{0:.4f}\n".format(model_r2[0], model_r2[1], model_err[0], model_err[1]))
            print("Class-specific Accuracies:\n\t")
            for k in range(n_classes):
                print("Class {}: {}% +/-{}; ".format(k, model_classwise_acc[str(k)][0], model_classwise_acc[str(k)][1]))
            print('\n')
            print("Class-specific Precisions:\n\t")
            for k in range(n_classes):
                print("Class {}: {}% +/-{}; ".format(k, model_classwise_precis[str(k)][0], model_classwise_precis[str(k)][1]))
            print('\n')
        print("Results: tp:{}, tn:{}, fp:{}, fn:{}\n".format(model_tp, model_tn, model_fp, model_fn))
        print("Time Elapsed: {0:.4f} average per run for total of {0:.4f}".format(np.mean(np.array(time_elapsed)), np.sum(np.array(time_elapsed))))
        print('\n\n')
        print()
    if binary: return model_acc[0], model_fscore[0]
    else: return model_acc[0], model_fscore[0], model_classwise_acc


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
    verbosity=True

    args = sys.argv
    filepath = args[1]
    model = args[2]
    binary = args[3]
    write_best = False

    if model == 'nn': fix_seeds()                 # fix the seeds for reproducibility

    X, encoded_y, gt_y, n_classes, n_feats = prep_data(filepath, binary=binary, modeltype=model, sep=';')

    #parameter options for models
    pows_C = [-2, -1, 0, 1, 2]
    pows_g = [-15, -10, -5, 0, 3]
    C = [2**k for k in pows_C]
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    poly = [2, 3, 4]
    gamma = [2**k for k in pows_g]
    e = [0, 0.01, 0.1, 0.5, 1, 2, 4]
    loss = ['hinge', 'squared_hinge']
    mult_class=['ovr', 'crammer_singer']

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

    if model   == 'svm':       param_grid = dict(C=C, kernel=kernel, degree=poly, gamma=gamma)
    elif model == 'svr':       param_grid = dict(C=C, kernel=kernel, degree=poly, gamma=gamma, ep=e)
    elif model == 'linearsvm': param_grid = dict(C=C, loss=loss, mult_class=mult_class)
    elif model == 'nn':        param_grid = dict(n_layers=n_layers, units1=units, units2=units, units3=units, activationh=activation, use_bias=use_bias, kernel_initializer=initializer, bias_initializer=initializer, batch_size=batch_size, optimizer=optimizer)

    param_combos = len(param_grid)

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

    if not binary:
        for k in range(n_classes):
            for rank in range(1,3):
                best_acc['class{}_rank{}_model_num'.format(k,rank)] = 0
                best_acc['class{}_rank{}_acc'.format(k,rank)] = 0
                best_acc['class{}_rank{}_params'.format(k,rank)] = {}
                write_scores(best_acc, best_fscore)

    # test model using default parameters
    if binary: acc_mean, fscore_mean = test_model(X, encoded_y, gt_y, n_feats, n_classes, binary=True, modeltype=model, verbose=verbosity)
    else: acc_mean, fscore_mean, class_acc = test_model(X, encoded_y, gt_y, n_feats, n_classes, binary=False, modeltype=model, verbose=verbosity)

    # generates random combination of parameters from the parameter grid
    # number of combinations initially set to max possible combinations
    # Note that training a model for each combinations would make this program
    # pretty much run indefinitely unless a massive distributed system is used.
    param_list = list(ParameterSampler(param_grid, n_iter=param_combos))

    # test the models that result from the combinations of parameters
    # Store the best accuracies and F1 Scores found
    for d in range(len(param_list)):
        if binary:
            acc_mean, fscore_mean = test_model(X, encoded_y, gt_y, n_feats, n_classes, binary=binary, modeltype=model, verbose=verbosity, param_dict=param_list[d], model_num=d)
        else:
            acc_mean, fscore_mean, class_acc = test_model(X, encoded_y, gt_y, n_feats, n_classes, binary=binary, modeltype=model, verbose=verbosity, param_dict=param_list[d], model_num=d)
        if write_best:
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
            if not binary:
                tie_class_acc=1
                tie_class_fscore=1
                for k in range(n_classes):
                    if class_acc[str(k)] > best_acc['class{}_rank1_acc'.format(k)]:
                        best_acc['class{}_rank1_model_num'.format(k)] = d
                        best_acc['class{}_rank1_acc'.format(k)] = class_acc[str(k)]
                        best_acc['class{}_rank1_params'.format(k)] = param_list[d]
                        write_scores(best_acc, best_fscore)
                    elif class_acc[str(k)] > best_acc['class{}_rank2_acc'.format(k)]:
                        best_acc['class{}_rank2_model_num'.format(k)] = d
                        best_acc['class{}_rank2_acc'.format(k)] = class_acc[str(k)]
                        best_acc['class{}_rank2_params'.format(k)] = param_list[d]
                        write_scores(best_acc, best_fscore)
                    elif class_acc[str(k)] == best_acc['class{}_rank2_acc'.format(k)]:
                        best_acc['class{}_rank2_{}_model_num'.format(k,tie_acc)] = d
                        best_acc['class{}_rank2_{}_acc'.format(k,tie_acc)] = class_acc[str(k)]
                        best_acc['class{}_rank2_{}_params'.format(k,tie_acc)] = param_list[d]
                        write_scores(best_acc, best_fscore)
                        tie_class_acc += 1


                    if class_fscore[str(k)] > best_fscore['class{}_rank1_fscore'.format(k)]:
                        best_fscore['class{}_rank1_model_num'.format(k)] = d
                        best_fscore['class{}_rank1_fscore'.format(k)] = class_fscore[str(k)]
                        best_fscore['class{}_rank1_params'.format(k)] = param_list[d]
                        write_scores(best_acc, best_fscore)
                    elif class_fscore[str(k)] > best_fscore['class{}_rank2_fscore'.format(k)]:
                        best_fscore['class{}_rank2_model_num'.format(k)] = d
                        best_fscore['class{}_rank2_fscore'.format(k)] = class_fscore[str(k)]
                        best_fscore['class{}_rank2_params'.format(k)] = param_list[d]
                        write_scores(best_acc, best_fscore)
                    elif class_fscore[str(k)] == best_fscore['class{}_rank2_fscore'.format(k)]:
                        best_fscore['class{}_rank2_{}_model_num'.format(k,tie_fscore)] = d
                        best_fscore['class{}_rank2_{}_fscore'.format(k,tie_fscore)] = model_fscore[str(k)][0]
                        best_fscore['class{}_rank2_{}_params'.format(k,tie_fscore)] = param_list[d]
                        write_scores(best_acc, best_fscore)
                        tie_class_fscore += 1
