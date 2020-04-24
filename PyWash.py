from methods.BandA.Normalization import normalize
#from src.BandA.OutlierDetector import identify_outliers, estimate_contamination
from methods.BandB.ptype.Ptype import Ptype
from methods.BandB.MissingValues import handle_missing
from methods.BandC.ParserUtil import assign_parser
#from src.BandC.Exports import *
#from src.Exceptions import *
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
        self.verbose = verbose
        self.file_path = file_path
        self.data = None
        self.parser = None
        self.score = None
        self.col_types = None
        # When a path to a file or the contents are given, parse the file and load the data
        if file_path is not None:
            self.parser = assign_parser(file_path=file_path, contents=contents, verbose=verbose)
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
        self.data = self.infer_data_types_ptype()

    def set_data(self, df):
        """ Sets an pre-parsed DataFrame as the data of the SharedDataFrame """
        self.data = df
        self.data = self.infer_data_types_ptype()

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
            return export_arff(self.name, self.data,
                               self.parser.attributes, self.parser.description)

    # Preview functions #####
    def returnPreview(self):
        """ Return a preview (5 rows) of the dataset """
        previewData = self.data.tail(5)
        return previewData

    # Data Cleaning functions #####
    def startCleaning(self, columnData, handleMissing, normalizationColumns, standardizationColumns, removeDuplicates):
        """ Main data cleaning function, use function defined below this one """
        print('changing column types')
        self.changeColumns(columnData)
        if removeDuplicates == True:
            print('removing duplicates')
            self.removeDuplicateRows()
        #Currently errors sometimes, rip
        if handleMissing == '1':
            print('handling missing data')
            #'remove' translates to: Jury-rig it to just drop the rows with NAs
            self.missing('remove', ['n/a', 'na', '--', '?']) #<- these are detected NA's, put in here :)
        if normalizationColumns is not None:
            print('handling normalization')
            self.normalizeColumns(normalizationColumns)
        if standardizationColumns is not None:
            print('handling standardization')
            self.standardizeColumns(standardizationColumns)
        print('done cleaning!')

    def changeColumns(self,columnData):
        """ Remove duplicate rows if selected in preview """
        #Check if columns need to be removed
        for item in self.data.columns:
            remove = True
            for item2 in columnData:
                if item == item2[0]:
                    remove = False
            if remove == True:
                self.data = self.data.drop(columns=item)
                print('column removed: '+str(item))
        #Now change the datatypes of the leftover columns
        for item in columnData:
            #Check if needs to be removed
            self.data = self.data.astype({item[0]:item[1]})
        print(self.data.dtypes)

    def removeDuplicateRows(self):
        print('Removing duplicates')
        rows = len(self.data.index)
        """ Remove columns selected for removal in preview """
        self.data.drop_duplicates(inplace=True)
        print('rows removed: '+str(rows - len(self.data.index)))
        return self.data

    def normalizeColumns(self,normalizationColumns):
        """ Normalize columns selected for removal in preview """
        scale_range='0,1'
        print(normalizationColumns)
        self.data = normalize(self.data, normalizationColumns, 'normalize', tuple(int(i) for i in scale_range.split(',')))
#        self.data = normalize(self.data, columns, setting, tuple(int(i) for i in scale_range.split(',')))
        return self.data

    def standardizeColumns(self,standardizationColumns):
        """ Standardize columns selected for removal in preview """
        scale_range='0,1'
        print(standardizationColumns)
        self.data = normalize(self.data, standardizationColumns, 'standardize', tuple(int(i) for i in scale_range.split(',')))
        return self.data

    # BandB functions #####
    def missing(self, setting, na_values):
        """ Fix the missing values of the dataset """
        self.data = handle_missing(self.data, setting, na_values)

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
        self.col_types = types_dct

        # try to change types of columns iteratively
        for item in types_dct.items():
            item = (item,)
            try:
                df = df.astype(dict(item))
            except:
                pass
        return df

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
