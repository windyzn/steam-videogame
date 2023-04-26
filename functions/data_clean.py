import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction import text 
import random

def preview_data(data):
    '''Preview data in markdown format'''
    return data.head().to_markdown(index=False)

def print_shape(data, data_name:str):
    '''Print the number of rows x columns of the data.'''
    print(f"The dataframe '{data_name}' has {data.shape[0]} rows and {data.shape[1]} columns.")

def strip_df(data):
    return data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

def check_duplicate_ids(data, id:list[str]):
    '''Check if there are any duplicates of the given ID'''
    duplicate_ids = data[data.duplicated(subset=[id], keep=False)][id].unique()

    if len(duplicate_ids) == 0:
        print(f"Data is clean, each row is a unique {id}.")
    else:
        print(f"Existing duplicate {id}: {duplicate_ids}")

def preview_top_x(data, col, x:int):
    '''Returns the top x games by col'''
    return data.sort_values(by=col, ascending=False)[['title', col]].head(x).to_markdown(index=False)

def preview_bottom_x(data, col, x:int):
    '''Returns the bottom x games by col'''
    return data.sort_values(by=col, ascending=True)[['title', col]].head(x).to_markdown(index=False)

def remove_upper_outliers(data, col, sd):
    '''Remove upper outliers greaer than x sd from the mean'''
    col_std = np.std(data[col])
    col_mean = np.mean(data[col])
    return data[data[col] < (col_mean + sd*col_std)]

def clean_missing(data: pd.DataFrame, id):
    '''Drop features if >20% missing, else drop IDs'''
    data_clean = data.copy()

    # Identify biomarkers (columns) with missing values
    missing_cols = data_clean.isnull().any()

    # Get the names of the biomarkers (columns) with missing values
    cols_to_drop = missing_cols[missing_cols].index

    # Count the number of missing values in each biomarker (column)
    n_missing_values = data_clean[cols_to_drop].isnull().sum()

    # Select only the biomarkers with more than 20% missing values
    cols_to_drop = n_missing_values[n_missing_values / data_clean.shape[0] > 0.2].index

    # Print the name and number of missing values for each biomarker (column)
    print("FEATURES WITH >20% MISSINGNESS")
    for column, value in n_missing_values.items():
        if column in cols_to_drop:
            print(f"{column} is being dropped with {value} missing values")

    # Print the number of biomarkers (columns) in the original DataFrame
    print()
    n_col_before = data_clean.shape[1]
    print(f"Before: {n_col_before} features")

    # Drop the identified biomarkers (columns)
    data_clean = data_clean.drop(cols_to_drop, axis=1)

    # Print the number of biomarkers (columns) in the resulting DataFrame
    n_col_after = data_clean.shape[1]
    print(f"After: {n_col_after} features")
    print(f"{n_col_before - n_col_after} features dropped")
    print()

    # Identify sample_ids (rows) with missing values
    missing_rows = data_clean.isnull().any(axis=1)

    # Get the indexes of the sample_ids (rows) with missing values
    rows_to_drop = missing_rows[missing_rows].index

    # Count the number of missing values in each sample_id (row)
    n_missing_values = data_clean.loc[rows_to_drop].isnull().sum(axis=1)

    # Print the name and number of missing values for each sample_id (row)
    print("DROP IDS WITH MISSINGNESS")
    for row in rows_to_drop:
        id = data_clean.loc[row, "id"]
        value = n_missing_values[row]
        print(f"{id} is being dropped with {value} missing values")

    # Print the number of sample_ids (rows) in the original DataFrame
    print()
    n_row_before = data_clean.shape[0]
    print(f"Before: {n_row_before} ids")

    # Drop the identified sample_ids (rows)
    data_clean = data_clean.drop(rows_to_drop, axis=0)

    # Print the number of sample_ids (rows) in the resulting DataFrame
    n_row_after = data_clean.shape[0]
    print(f"After: {n_row_after} ids")
    print(f"{n_row_before - n_row_after} ids dropped")
    print()

    # Check if the DataFrame still contains any invalid values
    if data_clean.isnull().any().any():
        print("Error: The DataFrame contains invalid values")
    else:
        print("The DataFrame is clean")
        print(
            f"Final data contains {n_col_after} features and {n_row_after} ids "
        )

    return pd.DataFrame(data_clean)

def preprocess_text(text):
    '''Returns preprocessed text'''
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove stop words and stem the tokens
    stemmed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]

    # Join the stemmed tokens back into a single string
    processed_text = ' '.join(stemmed_tokens)

    return processed_text

# Function to select a random value from a semi-colon separated string
def select_random_value(string):
    '''Select random entry in column with values separated by semicolon'''
    values = string.split(';')
    return random.choice(values)