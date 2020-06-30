import pandas as pd
import numpy as np
from scipy import stats
from Pywash2.methods.BandB.ptype.Ptype import Ptype
from Pywash2.methods.BandA.OutlierDetector import estimate_contamination, identify_outliers
import seaborn as sns
from pyod.models.knn import KNN as knn
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.mcd import MCD
from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from pyod.models.lscp import LSCP

def interpretable_values(df):
    '''
    Checks whether the data is documented or not.

    Parameters
    ----------
    df : the dataframe that needs analyzing.

    user_input : the user tells us whether the data has been documented, partially documented, or not documented.

    Returns
    -------
    quality_measure : The quality measure of documentation.
    '''
    ##TODO
    #userinput whether the data is documented, partially documented or not documented.

    if user_input == 'not documented':
        quality_measure = 0
    elif user_input == 'partially documented':
        quality_measure = 0.5
    if user_input == 'documented':
        quality_measure = 1
    return quality_measure

def feature_scaling(df):
    '''
    Performs a kolmogorov-smirnov on all the columns that were predicted to be numerical. Then calculates which
    percentage of these columns is normally distributed.

    Parameters
    ----------
    df : the dataframe that needs analyzing.


    Returns
    -------
    quality_measure : The percentage of normalized continuous columns.
    '''
    convert_dct = {'integer': 'int64', 'string': 'object', 'float': 'float64', 'boolean': 'bool',
                   'date-iso-8601': 'datetime64[ns]', 'date-eu': 'datetime64[ns]',
                   'date-non-std-subtype': 'datetime64[ns]', 'date-non-std': 'datetime64[ns]', 'gender': 'category',
                   'all-identical': 'category'}
    ptype = Ptype()
    ptype.run_inference(df)
    predicted = ptype.predicted_types
    count_normal_vars = 0
    count_continuous_vars = 0
    for key in predicted:
        # print(key, predicted[key])
        if predicted[key] == 'string' or predicted[key] == 'float':

            try:
                pd.to_numeric(df[key])
                count_continuous_vars += 1
                if stats.kstest(df[key], 'norm').pvalue <= 0.05:
                    continue
                else:
                    count_normal_vars += 1

                # null-hypothesis is no difference. p-value <= 0.05: not normal.
            except:
                print('Column {} could not be transformed to numeric.'.format(key))

    if count_continuous_vars > 0:
        quality_measure = count_normal_vars / count_continuous_vars * 100
    else:
        quality_measure = 1


    return quality_measure

def outlier_detection(df):
    '''
    Performs a contamination check and an outlier detection to see what percentage of the data is an outlier.

    Parameters
    ----------
    df : the dataframe that needs analyzing.


    Returns
    -------
    quality_measure : The percentage of outliers in the data.
    '''
    contamination = estimate_contamination(df)
    features = df.columns
    outliers = identify_outliers(df, features=features, contamination=contamination)[0]
    perc = outliers['prediction'].sum()/df.shape[0] * 100
    if perc <= 1:
        quality_measure = 1
    elif perc > 1 and perc <= 5:
        quality_measure = 0.75
    elif perc > 5 and perc <= 10:
        quality_measure = 0.5
    elif perc > 10 and perc <= 20:
        quality_measure = 0.25
    else:
        quality_measure = 0
    return quality_measure



# path = "C:/DataScience/ptype-datasets/main/main/data.gov/3397_1"
# df = pd.read_csv(path + '/data.csv')
# a = outlier_detection(df)
# print(a)

path = "C:/Users/20175848/Dropbox/Data Science Y3/Cognitive science"
df = pd.read_csv(path + '/rec_tracks.csv')
a = outlier_detection(df)
print(a)