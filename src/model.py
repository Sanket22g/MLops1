from sklearn.ensemble import RandomForestClassifier
import logging
import os
import pandas as pd

logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)
log_file = os.path.join(logs_dir, "model_training.log")
logger = logging.getLogger("model_training")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str):
    """
    Load data from a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file."""
    
    try:
        logger.info(f"Loading data from file: {file_path}")
        df = pd.read_csv(file_path)
        logger.debug(f"Data loaded successfully. Number of records: {len(df)} from {file_path}")
        return df
    except pd.errors.ParserError as pw:
        logger.error(f"Parsing error while reading the CSV file: {file_path}. Error: {pw}")
        raise
    except Exception as e:
        logger.error(f"Error during data loading from file: {file_path}. Error: {e}")
        raise

def train_model(X_train, y_train,parm: dict)-> RandomForestClassifier:
    """
    Train a RandomForestClassifier model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.

    Returns:
        RandomForestClassifier: Trained model.
    """
    try:
        logger.info("Starting model training")
        model = RandomForestClassifier(**parm)
        model.fit(X_train, y_train)
        logger.debug("Model training completed successfully")
        return model
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

def save_model(model: RandomForestClassifier, model_path: str) -> None:
    """
    Save the trained model to a file.

    Args:
        model (RandomForestClassifier): The trained model.
        model_path (str): The path to save the model file.
    """
    try:
        import joblib
        logger.info(f"Saving model to {model_path}")
        
        # Create directory if it doesn't exist
        model_dir = os.path.dirname(model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(model, model_path)
        logger.debug("Model saved successfully")
    except Exception as e:
        logger.error(f"Error saving the model: {e}")
        raise

def main():
    try:
        train_file_path = "vectorized_data/x_train_tfidf.csv"
        test_file_path = "vectorized_data/x_test_tfidf.csv"
        model_save_path = "models/random_forest_model.joblib"

        logger.info("Loading training data")
        train_df = load_data(train_file_path)
        logger.info("Loading testing data")
        test_df = load_data(test_file_path)

        X_train = train_df.drop("label", axis=1)
        y_train = train_df["label"]

        X_test = test_df.drop("label", axis=1)
        y_test = test_df["label"]

        model_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }

        model = train_model(X_train, y_train, model_params)

        save_model(model, model_save_path)

    except Exception as e:
        logger.error(f"Error in main function for model training: {e}")
        raise

if __name__ == "__main__":
    main()