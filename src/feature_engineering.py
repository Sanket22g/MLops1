import pandas as pd 
import logging 
import os
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer


logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)
log_file = os.path.join(logs_dir, "feature_engineering.log")
logger=logging.getLogger("feature_engineering")
logger.setLevel(logging.DEBUG)

counsoler_handler=logging.StreamHandler()
counsoler_handler.setLevel(logging.INFO)

file_handler=logging.FileHandler(log_file)

file_handler.setLevel(logging.DEBUG)
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
counsoler_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(counsoler_handler)
logger.addHandler(file_handler)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file."""
    
    try:
        logger.info(f"Loading data from file: {file_path}")
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug(f"Data loaded successfully. Number of records: {len(df)} from {file_path}")
        return df
    except pd.errors.ParserError as pw:
        logger.error(f"Parsing error while reading the CSV file: {file_path}. Error: {pw}")
        raise
    except Exception as e:
        logger.error(f"Error during data loading from file: {file_path}. Error: {e}")
        raise


def apply_tfidf_vectorization(train_data: pd.DataFrame, test_data: pd.DataFrame, text_column: str, max_features: int):
    """
    Apply TF-IDF vectorization to the text data in the specified column of the training and testing DataFrames.

    Args:
        train_data (pd.DataFrame): The training DataFrame.
        test_data (pd.DataFrame): The testing DataFrame.
        text_column (str): The name of the column containing text data to be vectorized.
        max_features (int): The maximum number of features to be extracted by the TF-IDF vectorizer.
    Returns:
        tuple: A tuple containing the TF-IDF vectorized training and testing data as sparse matrices.
    """
    try:
        logger.info(f"Applying TF-IDF vectorization to column: {text_column} with max_features: {max_features}")
        tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
        x_train=train_data["text"].fillna('').values
        x_test=test_data["text"].fillna('').values

        y_train=train_data["label"].values
        y_test=test_data["label"].values

        x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
        x_test_tfidf = tfidf_vectorizer.transform(x_test)

        x_train_df=pd.DataFrame(x_train_tfidf.toarray())
        x_train_df["label"]=y_train

        x_test_df=pd.DataFrame(x_test_tfidf.toarray())
        x_test_df["label"]=y_test
        logger.debug(f"TF-IDF vectorization completed successfully. Shape of training data: {x_train_df.shape}, Shape of testing data: {x_test_df.shape}")
        return x_train_df, x_test_df
    except KeyError as ke:
        logger.error(f"Key error during TF-IDF vectorization. Error: {ke}")
        raise
    except Exception as e:
        logger.error(f"Error during TF-IDF vectorization. Error: {e}")
        raise


def main():
    """
    Docstring for main
    :return: Description
    :rtype: None

    """
    try:
        # Load parameters
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
        
        feature_params = params['feature_engineering']
        max_features = feature_params['max_features'] 
        text_column = feature_params['text_column']
        
        logger.info("Starting main function for feature engineering")
        train_df=pd.read_csv("cleaned_data/train_data_cleaned.csv")
        test_df=pd.read_csv("cleaned_data/test_data_cleaned.csv")
        logger.debug(f"Data loaded successfully. Train records: {len(train_df)}, Test records: {len(test_df)}")

        x_train_df, x_test_df = apply_tfidf_vectorization(train_df, test_df, text_column=text_column, max_features=max_features)

        #store the vectorized data
        vectorized_data_dir = "vectorized_data"
        os.makedirs(vectorized_data_dir, exist_ok=True)
        x_train_df.to_csv(os.path.join(vectorized_data_dir, "x_train_tfidf.csv"), index=False)
        x_test_df.to_csv(os.path.join(vectorized_data_dir, "x_test_tfidf.csv"), index=False)
        logger.debug(f"Vectorized training and testing data saved successfully to {vectorized_data_dir} directory.")  

    except Exception as e:
        logger.error(f"Error in main function for feature engineering. Error: {e}")
        raise
    except FileExistsError as fee:
        logger.error(f"File already exists error in main function for feature engineering. Error: {fee}")
        raise
    except FileNotFoundError as fnfe:
        logger.error(f"File not found error in main function for feature engineering. Error: {fnfe}")
        raise
    except pd.errors.ParserError as pe:
        logger.error(f"Parsing error in main function for feature engineering. Error: {pe}")
        raise
if __name__ == "__main__":
    main()