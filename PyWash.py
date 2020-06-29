from methods.BandA.Normalization import normalize
from methods.BandA.OutlierDetector import identify_outliers, estimate_contamination, outlier_ensemble
from methods.BandB.ptype.Ptype import Ptype
#from methods.BandB.MissingValues import handle_missing
from methods.BandC.ParserUtil import assign_parser
from methods.BandC.Exports import export_csv, export_arff
from methods.MissingValueMethod.missing_values import *

from pandas.core.frame import DataFrame
import pandas as pd

supported_export_filetypes = ['csv', 'arff']


class SharedDataFrame:
    """ Shared DataFrame
    Main Abstract Data Type to store, process and use the data
    """

    def __init__(self, file_path: str = None, contents: str = None, df: DataFrame = None,
                 name: str = None, verbose: bool = False):
        """ Initializes the SharedDataFrame
        Can be given a path to a file to parse
         or a dataset as string needed to be parsed
         or a parsed DataFrame can be given to be used
        """

        # Event Logger
        with open('eventlog.txt', 'a') as file:
            string = 'Data with file path: ' + str(locals()['file_path']) + ' is uploaded' + '\n' + '\n'
            file.write(string)

        self.verbose = verbose
        self.file_path = file_path
        self.data = None
        self.parser = None
        self.score = None
        self.col_types = None
        self.anomalies = None
        self.missing_values = None
        self.accuracy_ptypes = None
        # When a path to a file or the contents are given, parse the file and load the data
        if file_path is not None:
            # Event Logger

            self.parser = assign_parser(file_path=file_path, contents=contents, verbose=verbose)

            with open('eventlog.txt', 'a') as file:
                string = 'Parser to read data is assigned' + '\n' + '\n'
                file.write(string)

            self._load_data()
            self.name = self.parser.name
        # When a DataFrame is given, set the DataFrame as the SharedDataFrame data
        elif df is not None:
            self.set_data(df)
        if name is not None:
            self.name = name

    #    def __repr__(self):
    #        # TODO, create representation
    #        NotImplementedError("Create")

    def __str__(self) -> str:
        # SharedDataFrames are represented by their file_name and the dimensions of the data
        return str(self.file_path) + " " + str(self.data.shape)

    def _load_data(self):
        self.data = self.parser.parse()
        self.col_types, self.anomalies, self.missing_values = self.infer_data_types_ptype()
        print(self.col_types)
        # Event Logger
        with open('eventlog.txt', 'a') as file:
            string = 'Column types of the data have been predicted using PType. Anomalies and Missing values have ' \
                     'been annotated \n \n'
            file.write(string)

    def set_data(self, df):
        """ Sets an pre-parsed DataFrame as the data of the SharedDataFrame """
        self.data = df
        self.col_types, self.anomalies, self.missing_values = self.infer_data_types_ptype()

        # Event Logger
        with open('eventlog.txt', 'a') as file:
            string = 'Column types of the data have been predicted using PType. Anomalies and Missing values have ' \
                     'been annotated \n \n'
            file.write(string)

    def remove(self, indices):
        self.data = self.data.drop(indices)

    def get_dataframe(self):
        return self.data

    def get_dtypes(self):
        return self.data.dtypes.apply(lambda x: x.name).to_dict()

    def update_dtypes(self, dtypes):
        try:
            self.data = self.data.astype(dtypes)
        except ValueError:
            print('failed updating dtypes')
            pass

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    def analyze_data(self):
        """ Determine band value of dataset """
        pass

    def get_datascore(self):
        """ Return the band value of the dataset """
        return self.score

    def export_string(self, file_type) -> str:
        """ Returns a downloadable string of the dataset with a specified file type
        :param file_type: The file type to save the dataset as
        :return: String dataset with download capacities
        """
        if file_type not in supported_export_filetypes:
            raise AttributeError(
                'Selected file type {} is not supported for exports'.format(file_type))
        elif file_type == 'csv':
            return export_csv(self.data)
        elif file_type == 'arff':
            print(self.name)
            print(self.parser.attributes)
            print(self.parser.description)
            return export_arff(self.name, self.data,
                               self.parser.attributes, self.parser.description)

    # Preview functions #####
    def returnPreview(self):
        """ Return a preview (5 rows) of the dataset """
        previewData = self.data.tail(5)
        return previewData

    # Data Cleaning functions #####
    def startCleaning(self, columnData, handleMissing, handleOutlier, normalizationColumns, standardizationColumns,
                      removeDuplicates):
        """ Main data cleaning function, use function defined below this one """

        #Remove all remaining anomalies
        print(self.anomalies)
#        anomalyList = self.anomalies.keys()
        for item in list(self.anomalies): #Every column
            print(item) #Column name
            self.replace_anomalies(item,self.anomalies[item])
        self.anomalies = {} #Empty anomalies, data is successfully edited afaik but some anomalies remain in list only when doing it in this step
        print('done with anomalies!')
        print('column types:')
        print(self.col_types)
        self.data = self.set_data_types()

        self.changeColumns(columnData)
        if removeDuplicates == True:
            print('removing duplicates')
            self.removeDuplicateRows()
            with open('eventlog.txt', 'a') as file:
                string = 'Duplicate rows have been removed' + '\n' + '\n'
                file.write(string)

        if handleMissing != None:
            ############################################## TODO: FIX THIS SHIT ##############################################
            # Currently errors sometimes, rip
            print('handling missing data')
            print(handleMissing)
            # 'remove' translates to: Jury-rig it to just drop the rows with NAs
            # OLD   self.missing('remove', ['n/a', 'na', '--', '?'])  # <- these are detected NA-s, put in here :)
            self.missing(handleMissing)
            #TODO add event logger once jonas has integrated missing value imputation

        if int(handleOutlier) > 0:
            print('handling outliers')
            self.handleOutliers(handleOutlier)

            with open('eventlog.txt', 'a') as file:
                if int(handleOutlier) == 1:
                    string = 'Outliers have been detected and marked using Ptype from Ceritli et al. '+ '\n' + '\n'
                else:
                    string = 'Outliers have been detected and removed using Ptype from Ceritli et al. '+ '\n' + '\n'

                file.write(string)

        if normalizationColumns is not None:
            print('handling normalization')
            self.normalizeColumns(normalizationColumns)

            with open('eventlog.txt', 'a') as file:
                string = 'The following columns have been normalized: ' + str(normalizationColumns) + '\n' + '\n'
                file.write(string)

        if standardizationColumns is not None:
            print('handling standardization')
            self.standardizeColumns(standardizationColumns)

            with open('eventlog.txt', 'a') as file:
                string = 'The following columns have been standardized: ' + str(standardizationColumns) + '\n' + '\n'
                file.write(string)

        print('done cleaning!')

    def changeColumns(self, columnData):
        """ Remove duplicate rows if selected in preview """
        # Check if columns need to be removed
        for item in self.data.columns:
            remove = True
            for item2 in columnData:
                if item == item2[0]:
                    remove = False
            if remove == True:
                self.data = self.data.drop(columns=item)
                print('column removed: ' + str(item))

    def removeDuplicateRows(self):
        print('Removing duplicates')
        rows = len(self.data.index)
        """ Remove columns selected for removal in preview """
        self.data.drop_duplicates(inplace=True)
        print('rows removed: ' + str(rows - len(self.data.index)))
        return self.data

    def handleOutliers(self, handleNum):
        #Slow method
        if handleNum == '1' or handleNum == '2':
            self.data = outlier_ensemble(self.data)

            if handleNum == '1':
                # drop all outlier columns except prediction, which is renamed to 'outlier' for clarity
                self.data = self.data.drop(['anomaly_score'], axis=1)
                self.data.rename(columns={'prediction': 'outlier'})
            if handleNum == '2':
                # Remove detected outliers, drop all outlier columns
                self.data = self.data[self.data.prediction != 1]
                self.data = self.data.drop(['anomaly_score', 'prediction'], axis=1)

        #Fast method
        if handleNum == '3' or handleNum == '4':
            contamination = estimate_contamination(self.data)
            setting = [0,7,9]
            self.data = self.outlier(setting,contamination)

            if handleNum == '3':
                # drop all outlier columns except prediction, which is renamed to 'outlier' for clarity
                self.data = self.data.drop(['anomaly_score'], axis=1)
                self.data.rename(columns={'prediction': 'outlier'})
            if handleNum == '4':
                # Remove detected outliers, drop all outlier columns
                self.data = self.data[self.data.prediction != 1]
                self.data = self.data.drop(['anomaly_score', 'prediction'], axis=1)


    def normalizeColumns(self, normalizationColumns):
        """ Normalize columns selected for removal in preview """
        scale_range = '0,1'
        print(normalizationColumns)
        self.data = normalize(self.data, normalizationColumns, 'normalize',
                              tuple(int(i) for i in scale_range.split(',')))
        #        self.data = normalize(self.data, columns, setting, tuple(int(i) for i in scale_range.split(',')))
        return self.data

    def standardizeColumns(self, standardizationColumns):
        """ Standardize columns selected for removal in preview """
        scale_range = '0,1'
        print(standardizationColumns)
        self.data = normalize(self.data, standardizationColumns, 'standardize',
                              tuple(int(i) for i in scale_range.split(',')))
        return self.data

    # BandB functions #####
#OLD:     def missing(self, setting, na_values):
    def missing(self, specified_column):
        ##OLD##
        #""" Fix the missing values of the dataset """
        #self.data = handle_missing(self.data, setting, na_values)

        ##NEW##
        #Code below from Jonas Niederle
        classification = True  # Specify whether the dataset concerns a regression or classification target variable

        target_column = self.data[specified_column]  # Extract the target_column, when implemented in the tool this must be done by the user (maybe in earlier steps already)
        d = self.data
        print(target_column)

        dat_pr, y, dum_enc, le, cols, dat_excl, dtype_dict, imp_arr = prepr_dat(d, specified_column)

        results, times, imputer_dict = test_estimators(dat_pr, y, dum_enc, classification=classification)

        fig, best_imputer = vis_imp_scores(results, save_fig=False, dataset=d)

        dat_imp = impute_dat(dat_pr, imputer_dict[best_imputer])

        self.data = restore_dat(dat_imp, y, dum_enc, le, cols, dat_excl, target_column, dtype_dict)

    def infer_data_types_ptype(self):
        """ Infer datatypes using ptype and apply the datatypes to the dataset"""

        df = self.data
        convert_dct = {'integer': 'int64', 'string': 'object', 'float': 'float64', 'boolean': 'bool',
                       'date-iso-8601': 'datetime64[ns]', 'date-eu': 'datetime64[ns]',
                       'date-non-std-subtype': 'datetime64[ns]', 'date-non-std': 'datetime64[ns]', 'gender': 'category',
                       'all-identical': 'category'}

        ptype = Ptype()
        ptype.run_inference(df)
        predicted = ptype.predicted_types
        types_lst = [convert_dct.get(_type) for _type in predicted.values()]
        types_dct = dict(zip(predicted.keys(), types_lst))
        anomalies = ptype.get_anomaly_predictions()
        anomalies = {k: v for k, v in anomalies.items() if v}  # remove empty lists
        missing_vals = ptype.get_missing_data_predictions()
        missing_vals = {k: v for k, v in missing_vals.items() if v}

        integer_cols = [k for k, v in predicted.items() if v == 'integer']

        # estimate accuracy
        accuracy_col = {k: v.max() for k, v in ptype.all_posteriors['demo'].items()}

        # estimate category:
        for col in integer_cols:
            if len(df[col].unique()) <= 10:
                types_dct[col] = 'category'
                accuracy_col[col] = 'unknown'  # change accuracy of prediction

        self.accuracy_ptypes = accuracy_col

        return types_dct, anomalies, missing_vals

    def set_data_types(self):
        print('try to set data types!')
        # try to change types of columns iteratively
        df = self.data
        for item in self.col_types.items():
            item = (item,)
            try:
                df = df.astype(dict(item))
            except:
                print(item)
                pass
        return df

    def remove_anomaly_prediction(self, column_name, items):
        print(items)
        for item in items:
            print('deleting item: ' + str(item))
            for i in range(0,len(self.anomalies[column_name])):
                print(self.anomalies[column_name][i])
                if item == self.anomalies[column_name][i]:
                    del self.anomalies[column_name][i]
                    print(self.anomalies[column_name])
                    print('deleted item ' + str(item))
                    break
        for item in self.anomalies:
            print(self.anomalies[item])
            if self.anomalies[item] == []:
                del self.anomalies[item]
                break #Can only delete item from one column at the same time, so we can only find one


    def replace_anomalies(self, column_name, items):
        self.data[column_name][self.data[column_name].isin(items)] = None   ###UNTESTED, old application had self.anomalies[column_name] instead of items and could only replace entire columns
        self.remove_anomaly_prediction(column_name, items)

    # BandA functions #####
    def scale(self, columns, setting, scale_range=(0, 1)):
        """ Normalize the dataset """
        self.data = normalize(self.data, columns, setting, tuple(int(i) for i in scale_range.split(',')))
        return self.data

    def outlier(self, setting, contamination):
        algorithms = ['Isolation Forest', 'Cluster-based Local Outlier Factor', 'Minimum Covariance Determinant (MCD)',
                      'Principal Component Analysis (PCA)', 'Angle-based Outlier Detector (ABOD)',
                      'Histogram-base Outlier Detection (HBOS)', 'K Nearest Neighbors (KNN)',
                      'Local Outlier Factor (LOF)',
                      'Feature Bagging', 'One-class SVM (OCSVM)']
        if pd.isnull(self.data).values.any():
            # TODO fix missing data with missing(features, Xy)?
            raise ValueError('fix missing data first')

        algorithms = [algorithms[i] for i in setting]
        df_sorted, df_styled = identify_outliers(self.data, self.data.columns, contamination=contamination,
                                                 algorithms=algorithms)
        return df_sorted

    def contamination(self):
        return estimate_contamination(self.data)
