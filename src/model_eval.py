from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report
import pandas as pd 
import logging 
import os 
import joblib
import json
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)

log_file = os.path.join(logs_dir, "model_evaluation.log")
logger = logging.getLogger("model_evl")
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

def load_model(model_path):
    """
    Load a trained model from a file.
    
    :param model_path: Path to the saved model file
    """
    try:
        logger.info("Model loading started")
        model = joblib.load(model_path)
        logger.debug(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}. Error: {e}")
        raise


def load_data(data_path:str):
    """
    Docstring for load_data
    
    :param data_path: Description
    :type data_path: str
    """

    try:
        logger.info("Data loading started")
        df = pd.read_csv(data_path)
        logger.debug(f"Data loaded successfully from {data_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from {data_path}. Error: {e}")
        raise


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return metrics.
    
    :param model: Trained model
    :param X_test: Test features
    :param y_test: Test labels
    """
    try:
        logger.info("Model evaluation started")
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        
        logger.debug("Classification Report:")
        report = classification_report(y_test, y_pred)
        logger.debug(f"\n{report}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'classification_report': report
        }
    except Exception as e:
        logger.error(f"Error during model evaluation. Error: {e}")
        raise


def save_metrics_to_json(metrics: dict, output_path: str) -> None:
    """
    Save evaluation metrics to a JSON file.
    
    :param metrics: Dictionary containing evaluation metrics
    :param output_path: Path to save the JSON file
    """
    try:
        logger.info(f"Saving metrics to {output_path}")
        
        # Create directory if it doesn't exist
        metrics_dir = os.path.dirname(output_path)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.debug("Metrics saved successfully")
    except Exception as e:
        logger.error(f"Error saving metrics to JSON. Error: {e}")
        raise


def main():
    """
    Main function to evaluate the trained model.
    """
    try:
        logger.info("Starting model evaluation process")
        
        # Load the trained model
        model_path = "models/random_forest_model.joblib"
        model = load_model(model_path)
        
        # Load test data
        test_data_path = "vectorized_data/x_test_tfidf.csv"
        test_df = load_data(test_data_path)
        
        # Split features and labels
        X_test = test_df.drop("label", axis=1)
        y_test = test_df["label"]
        
        # Evaluate the model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Save metrics to JSON file
        metrics_output_path = "metrics/evaluation_metrics.json"
        save_metrics_to_json(metrics, metrics_output_path)
        
        logger.info("Model evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main function for model evaluation. Error: {e}")
        raise


if __name__ == "__main__":
    main()

