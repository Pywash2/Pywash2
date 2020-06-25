import json
import time
import missingno as msno
import numpy as np
import pandas as pd
from plotly.express import bar
import plotly.figure_factory as ff
import plotly.graph_objects as go
from fancyimpute import IterativeSVD
from fancyimpute import SoftImpute
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from progress.bar import Bar
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


# Transforming datatypes for OpenML datasets
def create_dtype_dict(filename):
    # Create a dtype dictionary for a file exported from OpenMl
    with open(filename) as file:
        dat_info = json.load(file)

    dtype_list = [(x['name'], x['type']) for x in dat_info['features']]
    dtype_dict = {}

    for feature in dtype_list:
        if feature[1] == 'nominal':
            dtp = 'category'

        elif feature[1] == 'numeric':
            dtp = 'float64'

        else:
            print(feature[1] + " not allowed")
            return

        dtype_dict[feature[0]] = dtp

    return dtype_dict


def transform_dtypes(dat, filename):
    dtype_dict = create_dtype_dict(filename)
    return dat.astype(dtype_dict)


# Visualizing missing values

def col_frac_na(df):
    na_count = df.isna().apply(sum, axis=0)
    frac_na = (len(df) - na_count) / len(df)
    return frac_na


def plot_frac_missing(df):
    frac_missing = col_frac_na(df)
    fig = bar(x=frac_missing.index, y=frac_missing.values, title='Proportion observed for each variable',
              labels={'x': 'Variable', 'y': 'Proportion Observed'})
    fig.show()
    return fig


def corr_missing_plot(df):
    na_corr = df.isna().corr()
    x = list(na_corr.columns)
    y = list(na_corr.index)
    z = na_corr.to_numpy()

    mask = np.triu_indices_from(z)
    z[mask] = np.nan
    z_text = np.around(z, decimals=2).astype(str)
    z_text[mask] = ""
    z_text[z_text == 'nan'] = ""

    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text,
                                      showscale=True, hoverongaps=False)
    fig.update_layout(title="Correlation between missingness of different variables",
                      yaxis_title="Variable", xaxis_title="Variable", xaxis_gridcolor='white')
    fig.update_xaxes(side='bottom')
    fig.update_yaxes(autorange='reversed')
    fig.show()
    return fig


def plot_missing_corr_heatmap(df):
    ax = msno.heatmap(df)
    ax.set_title("Correlation between missingness of different variables", size=20)
    return ax


# Preprocessing dataset

def set_missing(col):
    # Set common missing value place holders of a column to a missing dtype
    place_holders = ['NA', 'na', 'nan', 'NAN', 'NaN', '?',
                     'None', 'NONE', 'none', np.nan]

    # Ensure no trailing spaces to prevent place holders not being recognized
    col = col.apply(lambda x: x.strip() if (type(x) == str) else x)

    col[col.isin(place_holders)] = np.nan
    return col


def repl_with_na(df):
    # Replace common missing value place holders the dataset to a missing dtype
    return df.apply(set_missing)


def drop_na_column(df):
    return df.dropna(axis='columns', how='all')


def exclude_list(dat, prop=0.75):
    # Generate list of columns with high proportion of unique values
    dat_cat = dat.select_dtypes(exclude='number')
    excl_list = [col for col in dat_cat.columns if len(dat_cat[col].unique()) / len(dat_cat) >= prop]
    excl_string = 'Features: ' + ', '.join(
        excl_list) + ' have a high number of unique values and will thus be excluded during imputation'

    if len(excl_list) > 0:
        print(excl_string)

    return excl_list


def create_categories(df_cat):
    # Function which creates a list of categories (to be used by CategoricalEncoder),
    # making sure 'nan' is always the last category
    cat_list = []
    for col in df_cat:
        cat_col = df_cat[col].unique()
        cat_col = np.delete(cat_col, np.argwhere(cat_col == 'nan'))
        cat_col = np.append(cat_col, 'nan')
        cat_list.append(cat_col)
    return cat_list


def category_dtype_to_str(col):
    # to be used in .apply
    if col.dtype.name == 'category':
        col.cat.categories = col.cat.categories.astype(str)
        return col

    return col


def encode_features(dat):
    # Encode all categorical features using one hot encoding

    dat_cat = dat.select_dtypes(exclude='number').astype(str)
    dat_num = dat.select_dtypes(include='number')
    cols = list(dat_cat.columns) + list(dat_num.columns)
    dat_num = np.asarray(dat_num)
    dat_cat = dat_cat.fillna('nan')
    cats = create_categories(dat_cat)
    dum_enc = OrdinalEncoder(categories=cats)
    dat_cat = dum_enc.fit_transform(dat_cat)
    return (dat_cat, dat_num, dum_enc, cols)


def set_cat_na(dat_cat, dum_enc):
    # Set encoded values corresponding to nan category to nan

    cats = dum_enc.categories_
    for i in range(len(cats)):
        val_nan = np.argwhere(cats[i] == 'nan')[0][0]
        dat_cat[np.where(dat_cat[:, i] == val_nan), i] = np.nan
        # print(val_nan)

    return dat_cat


def drop_target_na(dat, target_column):
    bool_mask = dat[target_column].isna()
    if sum(bool_mask) > 0:
        print(str(sum(bool_mask)) + " rows contain missing values in the target column, these rows will be dropped.")

        return dat[~bool_mask].reset_index(drop=True)

    return dat


def enc_target_column(y):
    # Encode labels of the target column
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return y_enc, le


def plot_missing_matrix(df):
    ax = msno.matrix(df)
    ax.set_title("Matrix showing missingness in dataset", size=25)
    return ax


def plot_missing(df):
    fig1 = plot_frac_missing(df)
    fig2 = corr_missing_plot(df)
    ax = plot_missing_matrix(df)
    return (fig1, fig2, ax)


def save_figure(ax, dataset):

    file_path = "vis/" + dataset[:-4] + "_missing_matrix.png"

    ax.get_figure().savefig(file_path, dpi=400, bbox_inches='tight')


# Execute visualization and prepreprocessing in correct order

def vis_na(dat, save_fig=True, dataset=None):
    # Visualize NA values before imputation
    dat_na = repl_with_na(dat)
    fig1, fig2, ax = plot_missing(dat_na)

    if save_fig:

        html_path_bar = "vis/" + dataset[:-4] + "_missing_bar.html"
        html_path_corr = "vis/" + dataset[:-4] + "_missing_corr.html"

        fig1.write_html(html_path_bar)
        fig2.write_html(html_path_corr)

        save_figure(ax, dataset)


def prepr_dat(dat, target_column, dataset_path):
    # Replace placeholders with np.nan
    dat = repl_with_na(dat)

    # TIJDELIJK, NOG WEGHALEN!!
    dat = transform_dtypes(dat, dataset_path[:-4] + ".json")

    # Drop columns containing only missing values
    dat = drop_na_column(dat)

    # Ensure categorical columns have categories encoded as str type (needed to prevent future errors)
    dat = dat.apply(category_dtype_to_str)

    dtype_dict = dict(dat.dtypes)

    # Drop rows with missing values in target column
    dat = drop_target_na(dat, target_column)

    # Split dataframe into columns to impute/to exclude
    excl_list = exclude_list(dat)  # Generate list of columns to exclude
    imp_cols = list(dat.columns)
    imp_cols.remove(target_column)  # Remove target column from features to impute

    for item in excl_list:
        imp_cols.remove(item)

    # Split data into part to impute and part to exclude
    dat_excl = dat[excl_list]
    dat_impute = dat[imp_cols]

    imp_arr = np.array(dat_impute.isna().any(axis=1))

    # Encode (categorical) features
    dat_cat, dat_num, dum_enc, cols = encode_features(dat_impute)

    # Restore the missing values to np.nan (instead of a dummy category)
    dat_cat = set_cat_na(dat_cat, dum_enc)

    # Concatenate the two datasets into one
    dat_pr = np.concatenate((dat_cat, dat_num), axis=1)

    y = np.array(dat[target_column])
    y, le = enc_target_column(y)
    return dat_pr, y, dum_enc, le, cols, dat_excl, dtype_dict, imp_arr


@ignore_warnings(category=ConvergenceWarning)
def imputation_score_regression(X, y, estimator_name, estimator, inductive=True):
    # Calculate the imputation score of an imputation estimator for a classification problem

    # List of basic regressors to use
    regressors = [
        ("3NN_regr", KNeighborsRegressor(n_neighbors=3)),
        ("DT_regr", DecisionTreeRegressor(max_depth=3)),
        ("Ridge_regr", Ridge())
    ]

    scores = dict.fromkeys([first[0] for first in regressors])

    for key in scores.keys():
        scores[key] = []

    # Create 5-fold split
    kf_strat = StratifiedKFold(n_splits=5, shuffle=True)
    splits = [split for split in kf_strat.split(X, y)]

    if not inductive:
        X = estimator.fit_transform(X, y)

    # Create a progress bar
    bar = Bar(estimator_name, max=len(splits))

    # Loop through folds
    for split in splits:
        X_train, X_test = X[split[0]], X[split[1]]
        y_train, y_test = y[split[0]], y[split[1]]

        # When inductive mode is be supported,
        if inductive:
            # Fit the imputation estimator and transform the training/test set
            X_train = estimator.fit_transform(X_train, y_train)
            X_test = estimator.transform(X_test)


        # Loop through benchmark regressors
        for name, regressor in regressors:
            # Fit the simple regressor

            regressor.fit(X_train, y_train)


            y_pred = regressor.predict(X_test)
            mae = -mean_absolute_error(y_test, y_pred)

            scores[name].append(mae)

        #print("split " + estimator_name + " finished")

        bar.next()

    bar.finish()

    return scores


@ignore_warnings(category=ConvergenceWarning)
def imputation_score_classification(X, y, estimator_name, estimator, inductive=True):
    # Calculate the imputation score of an imputation estimator for a classification problem

    # Check whether multiclass or binary
    multiclass = len(np.unique(y)) > 2
    if multiclass:
        print("Multiclass problem detected, ROC AUC OVO method will be used to score estimators")

    # List of basic classifiers to use
    classifiers = [
        ("naive_bayes", GaussianNB()),
        ("decision_tr", DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=3)),
        ("LDA", LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
        ("3NN", KNeighborsClassifier(n_neighbors=3))
    ]

    scores = dict.fromkeys([first[0] for first in classifiers])

    for key in scores.keys():
        scores[key] = []

    # Create 5-fold split
    kf_strat = StratifiedKFold(n_splits=5, shuffle=True)
    splits = [split for split in kf_strat.split(X, y)]

    if not inductive:
        X = estimator.fit_transform(X, y)

    # Create a progress bar
    bar = Bar(estimator_name, max=len(splits))

    # Loop through folds
    for split in splits:
        X_train, X_test = X[split[0]], X[split[1]]
        y_train, y_test = y[split[0]], y[split[1]]

        # When inductive mode is be supported,
        if inductive:
            # Fit the imputation estimator and transform the training/test set
            X_train = estimator.fit_transform(X_train, y_train)
            X_test = estimator.transform(X_test)

        # Loop through benchmark classifiers
        for name, classifier in classifiers:
            # Fit the simple classifier
            classifier.fit(X_train, y_train)

            if multiclass:
                # Predict probabilites (required for AUC)
                y_pred = classifier.predict_proba(X_test)

                # Calculate ROC AUC score using multiclass OVO method (not sensitive to class imbalance)
                roc_score = roc_auc_score(y_test, y_pred, multi_class='ovo', average='macro')

            else:
                y_pred = classifier.predict(X_test)
                roc_score = roc_auc_score(y_test, y_pred)
            # scores[name].append(roc_auc_score(y_test, y_pred, multi_class='ovo'))

            scores[name].append(roc_score)

        #print("split " + estimator_name + " finished")
        bar.next()

    bar.finish()

    return scores


# Test different imputation methods and visualize results

def create_mode_mean_imputer(X, dum_enc):
    # Create a single value imputer which imputes with the mode and mean for categorical and
    # numerical columns respectively


    # Check if the data contains only numerical columns
    if len(dum_enc.categories_) == 0:
        return SimpleImputer(strategy="mean")

    # Check if the data contains only categorical columns
    if len(dum_enc.categories_) == X.shape[1]:
        return SimpleImputer(strategy="most_fequent")

    cat_cols = slice(0, len(dum_enc.categories_))  # Slice of the categorical columns
    num_cols = slice(len(dum_enc.categories_), X.shape[1])  # Slice of the numerical columns

    mean_mode_imp = ColumnTransformer([("Mode_Imp", SimpleImputer(strategy='most_frequent'), cat_cols),
                                       ("Mean_Imp", SimpleImputer(strategy='mean'), num_cols)])

    return mean_mode_imp


def test_estimators(X, y, dum_enc, classification=True):
    ModeMeanImputer = create_mode_mean_imputer(X, dum_enc)

    # List with all imputation algorithms to test, in tuples of (name, estimator object, inductive)
    impute_estimators = [
        ("ModeMeanImputer", ModeMeanImputer, True),
        ("KNNImputer", KNNImputer(), True),
        ("Iter_BayesianRidge", IterativeImputer(estimator=BayesianRidge(), random_state=0), True),
        ("Iter_DecisionTree",
         IterativeImputer(estimator=DecisionTreeRegressor(max_features='sqrt', random_state=0), random_state=0), True),
        ("Iter_RF", IterativeImputer(estimator=RandomForestRegressor(n_estimators=100, random_state=0), random_state=0),
         True),
        ("Iter_ExtraTrees",
         IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=100, random_state=0), random_state=0), True),
        ("Iter_KNRegr", IterativeImputer(estimator=KNeighborsRegressor(n_neighbors=15), random_state=0), True),
        ("Iter_SVD", IterativeSVD(rank=min(min(X.shape) - 1, 10), verbose=False), False),
        ("SoftImpute", SoftImpute(verbose=False), False)
    ]

    imp_scores = {}
    times = {}
    if not classification:
        for estimator_name, impute_estimator, inductive in impute_estimators:
            time1 = time.time()
            imp_scores[estimator_name] = imputation_score_regression(X, y, estimator_name, impute_estimator, inductive)
            time2 = time.time()
            times[estimator_name] = time2 - time1
            #print(estimator_name + " finished, took " + str(round(time2 - time1, 1)) + " seconds")

    if classification:
        for estimator_name, impute_estimator, inductive in impute_estimators:
            time1 = time.time()
            imp_scores[estimator_name] = imputation_score_classification(X, y, estimator_name, impute_estimator,
                                                                         inductive)
            time2 = time.time()
            times[estimator_name] = time2 - time1
            #print(estimator_name + " finished, took " + str(round(time2 - time1, 1)) + " seconds")

    imputer_dict = {}
    for estimator_name, impute_estimator, inductive in impute_estimators:
        imputer_dict[estimator_name] = impute_estimator

    return imp_scores, times, imputer_dict


def imputation_score(score_dict):
    # Calculate the mean imputation score and mean standard devation
    scores = []
    sd_list = []
    for key in score_dict.keys():
        scores.extend(score_dict[key])
        sd_list.append(np.std(score_dict[key]))

    return (np.mean(scores), np.mean(sd_list))


def vis_imp_scores(scores_dict, save_fig=True, dataset=None):
    x = []
    y = []
    errors = []

    for key in scores_dict.keys():
        score, std = imputation_score(scores_dict[key])
        x.append(key)
        y.append(score)
        errors.append(std)

    max_idx = y.index(max(y))
    best_imputer = x[max_idx]
    x[max_idx] = "<b>" + x[max_idx] + "</b>"

    fig = bar(x=x, y=y, error_y=errors, title="Imputation score for each imputation method",
              labels={'x': 'Imputation method', 'y': 'Imputation score'})

    if save_fig:

        html_path = "vis/" + dataset[:-4] + "_imputation_scores.html"

        fig.write_html(html_path)

    fig.show()
    return fig, best_imputer


# Impute missing values and transform data back to original Pandas dataframe

def impute_dat(dat, imputer):
    # Impute and transform data using imputer
    return imputer.fit_transform(dat)


def round_cat_dat(dat, dum_enc):
    # Round the categorical columns to integers
    dat[:, 0:len(dum_enc.categories_)] = np.round(dat[:, 0:len(dum_enc.categories_)], 0)
    return dat


def allow_no_na(dat, dum_enc):
    # Make sure no value was imputed as being the nan category

    cats = dum_enc.categories_
    for i in range(len(cats)):
        dat_max = len(cats[i]) - 2  # Get the maximum allowed value (last category is always nan)
        dat[:, i] = np.clip(dat[:, i], 0, dat_max)

    return dat


def inverse_transform_labels(y, encoder):
    return encoder.inverse_transform(y)


def inverse_transform_dat(dat, dum_enc):
    # Inverse transform the encoded categorical columns
    # dat[:, 0:len(dum_enc.categories_)] = dum_enc.inverse_transform(dat[:, 0:len(dum_enc.categories_)])
    return dum_enc.inverse_transform(dat)


def restore_dat(dat, y, dum_enc, lab_enc, cols, dat_excl, target_column, dtype_dict):
    dat = round_cat_dat(dat, dum_enc)  # Round the values to allowed categories
    dat = allow_no_na(dat, dum_enc)  # Disallow any imputed values which correspond to an "nan" category

    dat_cat, dat_num = dat[:, 0:len(dum_enc.categories_)], dat[:, len(dum_enc.categories_):]

    if dat_cat.shape[1] != 0:  # Check if there exist any categorical columns
        dat_cat = inverse_transform_dat(dat_cat, dum_enc)  # Inverse transform data

    y = inverse_transform_labels(y, lab_enc)  # Inverse transform y labels

    imp_df = pd.DataFrame(np.concatenate((dat_cat, dat_num), axis=1), columns=cols)

    df = pd.concat([imp_df, dat_excl], axis=1)
    df[target_column] = y
    df = df.astype(dtype_dict)
    return df


# Visualize imputed data using a parallel coordinates plot

def vis_encode_col(col):
    if not is_numeric_dtype(col.dtype):
        enc = OrdinalEncoder()
        return enc.fit_transform(np.array(col).reshape((len(col), 1))).flatten()
    return col


def vis_encode_cols(df):
    return df.apply(vis_encode_col, axis=0)


def pc_imputed(df, imp_arr, df_excl, save_fig=True, dataset=None):
    # Requires column named imputed

    subset = list(df.columns)
    for item in list(df_excl.columns):
        subset.remove(item)

    df = df[subset]
    df = vis_encode_cols(df.copy())
    imp_arr_num = np.zeros(len(imp_arr))
    imp_arr_num[imp_arr] = 1
    df['imputed'] = imp_arr_num
    dim = []
    for col in df.columns[:-1]:
        col_dict = {}
        col_dict['range'] = [min(df[col]), max(df[col])]
        col_dict['label'] = col
        col_dict['values'] = df[col]
        dim.append(col_dict)

    fig = go.Figure(data=
    go.Parcoords(
        line=dict(color=df['imputed'],
                  colorscale="Bluered",
                  colorbar={'tickvals': [0, 1], 'ticktext': ["not imputed", "imputed"]}),
        dimensions=dim
    )
    )
    fig.update_layout(title="PC plot visualizing the imputed datapoints")

    if save_fig:
        html_path = "vis/" + dataset[:-4] + "_parallel_coordinates.html"

        fig.write_html(html_path)

    fig.show()
