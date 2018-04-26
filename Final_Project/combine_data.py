import pandas as pd
import numpy as np

white = pd.read_csv('./white/winequality-white.csv', sep=';')
red = pd.read_csv('./red/winequality-red.csv', sep=';')

white['type'] = pd.Series(np.zeros(white.shape[0], dtype=int))
red['type'] = pd.Series(np.ones(red.shape[0], dtype=int))

wines = [white,red]

combo_type = pd.concat(wines, ignore_index=True)


debug=False
if debug:
    print(combo_type)
    print(white.shape[0] + red.shape[0])

cols = list(combo_type.columns.values)
cols.pop(cols.index('quality'))
cols.pop(cols.index('type'))
combo_quality = combo_type[cols+['type','quality']]

if debug:
    print(combo_quality)
    print(combo_quality.shape)

combo_type.to_csv('./combo-type/winetypecomparison.csv')
combo_quality.to_csv('./combo-quality/winequalitybytype.csv')
