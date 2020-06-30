import pandas as pd
from Pywash2.methods.BandB.ptype.Ptype import Ptype
from Pywash2.methods.BandB.check_categorical_distr import check_distr
import numpy as np

def column_types_measure(df):
    '''
    Checks whether the are predicted correctly and thus if the columns have the correct type.

    Parameters
    ----------
    df : the dataframe that needs analyzing.

    user_input : verification from the user if the columns were recognized correctly.

    Returns
    -------
    quality_measure : The percentage of correctly recognized columns.
    '''

    convert_dct = {'integer': 'int64', 'string': 'object', 'float': 'float64', 'boolean': 'bool',
                   'date-iso-8601': 'datetime64[ns]', 'date-eu': 'datetime64[ns]',
                   'date-non-std-subtype': 'datetime64[ns]', 'date-non-std': 'datetime64[ns]', 'gender': 'category',
                   'all-identical': 'category'}
    ptype = Ptype()
    ptype.run_inference(df)
    predicted = ptype.predicted_types

    ##TODO
    # get user input to verify which columns were predicted correctly
    #quality_measure = number_correctly_classfied/df.shape[1] * 100

    return quality_measure

def set_missing(col):

    # Set common missing value place holders of a column to a missing dtype
    place_holders = ['NA', 'na', 'nan', 'NAN', 'NaN', '?',
                     'None', 'NONE', 'none', np.nan]
    col[col.isin(place_holders)] = np.nan
    return col


def repl_with_na(df):
    # Replace common missing value place holders the dataset to a missing dtype
    return df.apply(set_missing)


def missing_values_measure(df):
    '''
    Checks how many missing values are present in the data and defines a quality measure for missing values.

    Parameters
    ----------
    df : the dataframe that needs analyzing.
    Returns
    -------
    quality_measure : A quality measure depending on the amount of missing values in the dataset.
    '''

    # print(df.isna().sum().sum())
    df = repl_with_na(df)
    # print(df.isna().sum().sum())

    perc_na = df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    if perc_na < 5:
        quality_measure = 1
    elif perc_na <= 10 and perc_na >= 5:
        quality_measure = 0.5
    elif perc_na <= 25 and perc_na > 10:
        quality_measure = 0.25
    else:
        quality_measure = 0
    return quality_measure

def inconsistent_entries_measure(df):
    '''
    Checks whether the entries follow the right distribution.

    Parameters
    ----------
    df : the dataframe that needs analyzing.

    user_input : User needs to verify if the distribution of the catagorical variables could be right or is off.

    Returns
    -------
    quality_measure : A quality measure depending on the amount of wrong entries in the dataframe.
    '''
    lst = check_distr(df)

    ##TODO
    # User needs to verify whether the amount of unique entries in lst could be right.

    # quality_measure = correct_columns / df.shape[1] * 100


    return quality_measure

def duplicate_records_measure(df):
    '''
    Checks whether there are duplicate records.

    Parameters
    ----------
    df : the dataframe that needs analyzing.
    Returns
    -------
    quality_measure : The percentage duplicate rows in the dataframe.
    '''

    duplicates = len(df[df.duplicated()])
    perc_dup = duplicates / len(df) * 100

    if perc_dup < 5:
        quality_measure = 1
    elif perc_dup <= 10 and perc_na >= 5:
        quality_measure = 0.5
    elif perc_dup <= 25 and perc_na > 10:
        quality_measure = 0.25
    else:
        quality_measure = 0
    return quality_measure

def meaningful_values_measure(df):
    '''
    Checks whether columnstypes are logical in terms of what they should represent.

    Parameters
    ----------
    df : the dataframe that needs analyzing.

    user_input : the user should answer whether the representation of the variable is useful as it is presented in the
                 dataframe.
    Returns
    -------
    quality_measure : The percentage of meaningful columns.
    '''

    ##TODO
    # requires user input to say something about if the values that are in the columns are meaningful. Yes/No will
    # suffice.

    quality_measure = number_useful_columns / df.shape[1] * 100
    return quality_measure

def quality_band_B(df):
    '''
    Performs all quality measures of band B in one function.
    Parameters
    ----------
    df : Dataframe that needs to be checked.
    file_path : path to the dataframe that needs to be checked.

    Returns
    -------
    out_df : Quality measures of band B in a DataFrame format.
    '''
    columns = column_types_measure(df)
    missing = missing_values_measure(df)
    inconsistent = inconsistent_entries_measure(df)
    duplicate = duplicate_records_measure(df)
    meaningful = meaningful_values_measure(df)

    output_lst = [columns, missing, inconsistent, duplicate, meaningful]
    index = ['columns', 'missing', 'inconsistent', 'duplicate', 'meaningful']

    out_df = pd.DataFrame(output_lst, index=index, columns=['Measures'])
    return out_df


path = "C:/DataScience/ptype-datasets/main/main/data.gov/3397_1"
df = pd.read_csv(path + '/data.csv')
a = quality_band_B(df)
print(a)