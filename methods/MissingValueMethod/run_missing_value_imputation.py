import pandas as pd
import pickle
import json
from missing_values import vis_na
from missing_values import prepr_dat
from missing_values import test_estimators
from missing_values import vis_imp_scores
from missing_values import impute_dat
from missing_values import restore_dat
from missing_values import pc_imputed

# Set the path of the (OpenML) dataset
folder = "test_datasets/CC-18"
dataset = "credit-approval.csv"
dataset_path = folder + "/" + dataset
classification = True  # Specify whether the dataset concerns a regression or classification target variable


# Function which can be used to store scores of the different imputation algorithms
def pickle_result(result, dataset):
    with open("results/" + dataset[:-4] + ".p", "wb") as file:
        pickle.dump(result, file)


# Load info about dataset, stored in a JSON file downloaded from OpenML (needed to specify the dtypes of the
# different columns), when implemented in the tool this is not necessary as dtype specification is done in earlier steps
with open(dataset_path[:-4] + ".json") as file:
    dat_info = json.load(file)

# Extract the target_column, when implemented in the tool this must be done by the user (maybe in earlier steps already)
target_column = dat_info['default_target_attribute']

# Read data, when implemented in the tool this not necessary
dat = pd.read_csv(dataset_path)

# Visualize missing values
vis_na(dat, save_fig=True, dataset=dataset)

# Preprocess data and split store encoder object/dtype dictionary needed to restore dataset later on
dat_pr, y, dum_enc, le, cols, dat_excl, dtype_dict, imp_arr = prepr_dat(dat, target_column, dataset_path)

# Test different estimators
results, times, imputer_dict = test_estimators(dat_pr, y, dum_enc, classification=classification)

# If desired results can be stored in a pickle file
# pickle_result([results, times], dataset)

# Visualize results
fig, best_imputer = vis_imp_scores(results, save_fig=True, dataset=dataset)

# Impute data, using best_imputer, possibly let the user select a different method if desired
dat_imp = impute_dat(dat_pr, imputer_dict[best_imputer])

imputed_df = restore_dat(dat_imp, y, dum_enc, le, cols, dat_excl, target_column, dtype_dict)

# Visualize imputed data using parallel coordinates plot
pc_imputed(imputed_df, imp_arr, dat_excl, save_fig=True, dataset=dataset)

print("Finished " + dataset)
