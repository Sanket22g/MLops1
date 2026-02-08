import logging 
import pandas as pd
import os
import string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')
ps = PorterStemmer()

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


def transform_data(text: str) -> str:
    """
    Transform a single text string by cleaning, removing stopwords, and stemming."""
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

def preprocess_data(df, text_column ="text", target="label") -> pd.DataFrame:

    """
    Preprocess the DataFrame by handling missing values.
    This funtion take three parameters, the dataframe, the text column and the target column. It will return the cleaned dataframe."""

    try  :
        logger.info("starting data preprocessing")
        df_cleaned = df.dropna()
        df_cleaned[text_column] = df_cleaned[text_column].apply(transform_data)
        le = LabelEncoder()
        df_cleaned[target] = le.fit_transform(df_cleaned[target])
        logger.debug(f"Data preprocessing completed successfully. Number of records after cleaning: {len(df_cleaned)}")
        return df_cleaned
    except KeyError as ke:
        logger.error(f"Key error during data preprocessing. Error: {ke}")
        raise
    except Exception as e:
        logger.error(f"Error during data preprocessing. Error: {e}")
        raise

def main(text_column="text", target="label") -> pd.DataFrame:
    """
    Docstring for main
    
    :param text_column: Description
    :type text_column: str
    :param target: Description
    :type target: str
    :return: Description
    :rtype: DataFrame
    """

    try:
        logger.info("Starting main function for data preprocessing")
        train_df=pd.read_csv("raw_data/train_data.csv")
        test_df=pd.read_csv("raw_data/test_data.csv")
        logger.debug(f"Data loaded successfully. Train records: {len(train_df)}, Test records: {len(test_df)}")

        logger.info("Preprocessing training data")
        train_df_cleaned = preprocess_data(train_df, text_column, target)
        test_df_cleaned = preprocess_data(test_df, text_column, target)
        logger.debug(f"Data preprocessing completed for training and testing data. Train records after cleaning: {len(train_df_cleaned)}, Test records after cleaning: {len(test_df_cleaned)}")

        #store the cleaned data
        cleaned_data_dir = "cleaned_data"
        os.makedirs(cleaned_data_dir, exist_ok=True)
        train_df_cleaned.to_csv(os.path.join(cleaned_data_dir, "train_data_cleaned.csv"), index=False)
        test_df_cleaned.to_csv(os.path.join(cleaned_data_dir, "test_data_cleaned.csv"), index=False)
        logger.debug(f"Cleaned training and testing data saved successfully to {cleaned_data_dir} directory.")  

    except Exception as e:
        logger.error(f"Error in main function for data preprocessing. Error: {e}")
        raise
    except FileExistsError as fee:
        logger.error(f"File already exists error in main function for data preprocessing. Error: {fee}")
        raise
    except FileNotFoundError as fnfe:
        logger.error(f"File not found error in main function for data preprocessing. Error: {fnfe}")
        raise
    except pd.errors.ParserError as pe:
        logger.error(f"Parsing error in main function for data preprocessing. Error: {pe}")
        raise

if __name__ == "__main__":
    main()











    

