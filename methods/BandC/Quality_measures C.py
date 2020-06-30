import pandas as pd
from Pywash2.methods.BandC.ParserUtil import assign_parser
from Pywash2.methods.BandB.ptype.Ptype import Ptype

def parse_ability_measure(df):
    '''Checks whether the application read the data appropriately in terms of parsing

    Parameters
    ----------

    df : DataFrame
        DataFrame containing the data.

    Returns
    -------
    quality_measure : The quality measure of parse-ability as an integer
    '''

    if not df.empty:
        quality_measure = 1
        print('Quality measurement for parse-ablity = 1')
    else:
        quality_measure = 0
        print('Quality measurement for parse-ablity = 0')
    return quality_measure


def data_storage_measure(df):
    '''
    Checks whether the application can perform algorithms on given data.

    Parameters
    ----------

    df : DataFrame containing the data.

    Returns
    -------
    quality_measure : The quality measure of storage as an integer
    '''
    try:
        ##TODO
        heftigalgoritme()
        quality_measure = 1
    except:
        print('The volume of the data is too big for our algorithms.')
        quality_measure = 0
    return quality_measure


def encoding_measure(file_path):
    '''
    Checks whether the automated encoding detection works on the given dataset.

    Parameters
    ----------

    File path : Path of the data set location.

    Returns
    -------
    quality_measure : Quality measure of encoding, either known encoding, or not.
    '''
    test = assign_parser(file_path)

    if test == None:
        quality_measure = 0
    else:
        quality_measure = 1
    return (quality_measure)


def unexpected_values_in_column(df):
    '''
    Decides the type of the columns and checks whether there are unexpected types.
    Parameters
    ----------
    df : Dataframe that needs to be checked.

    Returns
    -------
    errors : The amount of values that do not have the correct type in the dataframe.
    '''

    convert_dct = {'integer': 'int64', 'string': 'object', 'float': 'float64', 'boolean': 'bool',
                   'date-iso-8601': 'datetime64[ns]', 'date-eu': 'datetime64[ns]',
                   'date-non-std-subtype': 'datetime64[ns]', 'date-non-std': 'datetime64[ns]', 'gender': 'category',
                   'all-identical': 'category'}
    ptype = Ptype()
    ptype.run_inference(df)
    predicted = ptype.predicted_types
    types_lst = [convert_dct.get(_type) for _type in predicted.values()]
    types_dct = dict(zip(predicted.keys(), types_lst))
    not_as_expected = ptype.get_anomaly_predictions()
    missing_vals = ptype.get_missing_data_predictions()
    errors = 0
    for key in not_as_expected:
        errors += (len(not_as_expected[key]))
    return errors


def data_formats_measure(df):
    '''
    Checks whether the data formatting is done appropriately and calculates a score.

    Parameters
    ----------

    df : Dataframe containing the data

    Returns
    -------
    quality_measure : Quality measure in terms of data formatting in the form of an integer.
    '''
    unexpected = unexpected_values_in_column(df)
    certainty = 1  # The algorithm has an accuracy of above 99%.
    total_number = df.shape[1] * df.shape[0]
    number_expected = total_number - unexpected
    quality_measure = certainty * (number_expected / total_number)
    return quality_measure


def disjoint_datasets_measure():
    '''
    Checks whether the dataset is dis-joint. (never the case with PyWash, so measure = 1)

    Parameters
    ----------

    Returns
    -------
    quality_measure : Quality measure in terms of data formatting in the form of an integer.
    '''
    quality_measure = 1
    return quality_measure


def quality_band_C(df, file_path):
    '''
    Performs all quality measures of band C in one function.
    Parameters
    ----------
    df : Dataframe that needs to be checked.
    file_path : path to the dataframe that needs to be checked.

    Returns
    -------
    out_df : Quality measures of band C in a DataFrame format.
    '''
    parse = parse_ability_measure(df)
    storage = data_storage_measure(df)
    encoding = encoding_measure(file_path)
    format = data_formats_measure(df)
    disjoint = disjoint_datasets_measure()

    output_lst = [parse, storage, encoding, format, disjoint]
    index = ['parsing', 'storage', 'encoding', 'formatting', 'disjoint']

    out_df = pd.DataFrame(output_lst, index = index, columns=['Measures'])
    return out_df



# path = "C:/DataScience/ptype-datasets/main/main/data.gov/3397_1"
# df = pd.read_csv(path + '/data.csv')
#
# a = quality_band_C(df, path + '/data.csv')
# print(a)