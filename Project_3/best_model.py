import p3_classification as p3
import pandas as pd
import ast
"""
__author__ = "Jeremy Shi and Christian McDaniel"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Jeremy Shi and Christian McDaniel"
__email__ = "jeremy.shi@uga.edu, clm121@uga.edu"

This file reads in the parameters that provided the highest accuracy from
p3_classification.py and retrains/tests the model in p3_classification.py using
these parameters. The results are
"""
if __name__ == '__main__':

    best_list = pd.read_csv('best_scores.csv')

    best_params = best_list.iloc[1,1]

    params_dict = ast.literal_eval(best_params)
    p3.fix_seeds()
    X, encoded_y = p3.prep_data("./data/caravan.csv")
    for test in range(20):
        p3.test_model(X, encoded_y, params_dict, model_num=-10)
