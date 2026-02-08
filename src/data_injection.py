import logging 
import pandas as pd
import os 
from sklearn.model_selection import train_test_split

logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)
log_file = os.path.join(logs_dir, "data_injection.log")

logger=logging.getLogger("data_injection")
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



def data_save (url: str) -> pd.DataFrame:
    """
    Inject data from a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.
    Returns:
        pd.DataFrame: The DataFrame containing the injected data.   
        https://raw.githubusercontent.com/Sanket22g/dataset/refs/heads/main/reddit_artist_posts_sentiment.csv    
    """
    try:
        logger.info(f"Starting data injection from URL: {url}")
        df = pd.read_csv(url)
        logger.debug(f"Data injection completed successfully. Number of records: {len(df)} from {url} ")
        return df
    except pd.errors.ParserError as pw:
        logger.error(f"Parsing error while reading the CSV file from URL: {url}. Error: {pw}")
        raise
    except Exception as e:
        logger.error(f"Error during data injection from URL: {url}. Error: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the DataFrame by handling missing values."""

    try :
        logger.info("starting data preprocessing")
        df_cleaned = df.dropna()
        logger.debug(f"Data preprocessing completed successfully. Number of records after cleaning: {len(df_cleaned)}")
        return df_cleaned
    except Exception as e:
        logger.error(f"Error during data preprocessing. Error: {e}")
        raise

def save_data_to_csv(train_data: pd.DataFrame, test_data: pd.DataFrame, path:str) -> None:
    """
    Save the training and testing DataFrames to CSV files.

    Args:
        train_data (pd.DataFrame): The training DataFrame.
        test_data (pd.DataFrame): The testing DataFrame.
        train_path (str): The file path to save the training data CSV.
        test_path (str): The file path to save the testing data CSV.
    """
    try:
        raw_data_dir = "raw_data"
        os.makedirs(raw_data_dir, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_dir, "train_data.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_dir, "test_data.csv"), index=False)
        logger.info(f"Training and testing data saved successfully to {raw_data_dir} directory.")
    except Exception as e:
        logger.error(f"Error saving data to CSV files. Error: {e}")
        raise

def main():
    try:
        logger.info("Data injection process started.")
        url = "https://raw.githubusercontent.com/Sanket22g/dataset/refs/heads/main/reddit_artist_posts_sentiment.csv"
        df = data_save(url) 
        df_cleaned = preprocess_data(df)
        train_data, test_data = train_test_split(df_cleaned, test_size=0.2, random_state=42)
        save_data_to_csv(train_data, test_data, path="raw_data")
        logger.info("Data injection process completed successfully.")
    except Exception as e:
        logger.error(f"Data injection process failed. Error: {e}")
        raise

if __name__ == "__main__":
    main()