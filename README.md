# MLops1

In this project we build an end-to-end ML pipeline using DVC, Amazon S3, and Git. We analyze how the pipeline works and experiment with different approaches to improve it.

## Project Structure

```
MLops1/
├── src/
│   ├── data_injection.py
│   ├── pre_processing.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── model_eval.py
├── raw_data/
│   ├── train_data.csv
│   └── test_data.csv
├── cleaned_data/
├── vectorized_data/
├── models/
├── metrics/
├── logs/
├── .gitignore
├── dvc.yaml
└── README.md
```

## Building DVC Pipeline - Step by Step

### Step 1: Create GitHub Repository and Initial Setup

```bash
# Create a GitHub repo and clone it locally
git clone <your-repo-url>
cd MLops1

# Add experiments branch
git checkout -b experiments
```

### Step 2: Set Up Project Structure

```bash
# Create src folder with all components
mkdir src
# Add your Python scripts: data_injection.py, pre_processing.py, 
# feature_engineering.py, model.py, model_eval.py

# Create necessary directories
mkdir data models reports cleaned_data vectorized_data metrics logs
```

### Step 3: Configure .gitignore

Add the following to `.gitignore`:

```gitignore
# Data directories
data/
cleaned_data/
vectorized_data/
raw_data/

# Model files
models/

# Reports and metrics
reports/
metrics/

# Logs
logs/

# DVC
.dvc/cache
.dvc/tmp

# Python
__pycache__/
*.pyc
.venv/
venv/
*.egg-info/
```

### Step 4: Initial Git Commit

```bash
git add .
git commit -m "Initial project setup with src components"
git push origin main
```

### Step 5: Initialize DVC

```bash
# Initialize DVC in your project
dvc init

# Add remote storage (Amazon S3)
dvc remote add -d myremote s3://your-bucket-name/path

# Commit DVC initialization
git add .dvc .dvcignore
git commit -m "Initialize DVC"
git push
```

### Step 6: Create DVC Pipeline (dvc.yaml)

Create `dvc.yaml` file with stages:

```yaml
stages:
  data_preprocessing:
    cmd: python src/pre_processing.py
    deps:
      - src/pre_processing.py
      - raw_data/train_data.csv
      - raw_data/test_data.csv
    outs:
      - cleaned_data/train_data_cleaned.csv
      - cleaned_data/test_data_cleaned.csv

  feature_engineering:
    cmd: python src/feature_engineering.py
    deps:
      - src/feature_engineering.py
      - cleaned_data/train_data_cleaned.csv
      - cleaned_data/test_data_cleaned.csv
    outs:
      - vectorized_data/x_train_tfidf.csv
      - vectorized_data/x_test_tfidf.csv

  model_training:
    cmd: python src/model.py
    deps:
      - src/model.py
      - vectorized_data/x_train_tfidf.csv
      - vectorized_data/x_test_tfidf.csv
    outs:
      - models/random_forest_model.joblib

  model_evaluation:
    cmd: python src/model_eval.py
    deps:
      - src/model_eval.py
      - models/random_forest_model.joblib
      - vectorized_data/x_test_tfidf.csv
    metrics:
      - metrics/evaluation_metrics.json:
          cache: false
```

### Step 7: Run DVC Pipeline

```bash
# Test pipeline automation
dvc repro

# Check pipeline visualization
dvc dag
```

### Step 8: Commit Pipeline Configuration

```bash
git add dvc.yaml dvc.lock
git commit -m "Add DVC pipeline configuration"
git push
```

### Step 9: Track Data with DVC (Optional)

```bash
# Track large data files with DVC
dvc add raw_data/train_data.csv
dvc add raw_data/test_data.csv

# Push data to remote storage
dvc push

# Commit .dvc files
git add raw_data/.gitignore raw_data/*.dvc
git commit -m "Track data files with DVC"
git push
```

## DVC Pipeline Visualization

```
┌────────────────────┐
│ data_preprocessing │
└──────────┬─────────┘
           │
           v
┌─────────────────────┐
│ feature_engineering │
└──────────┬──────────┘
           │
           v
┌──────────────────┐
│ model_training   │
└──────────┬───────┘
           │
           v
┌──────────────────┐
│ model_evaluation │
└──────────────────┘
```

## Useful DVC Commands

```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro model_training

# Show pipeline DAG
dvc dag

# Show metrics
dvc metrics show

# Compare experiments
dvc metrics diff

# Pull data from remote
dvc pull

# Push data to remote
dvc push

# Check pipeline status
dvc status
```

## Technologies Used

- **DVC**: Data Version Control and pipeline automation
- **Git**: Code version control
- **Amazon S3**: Remote data storage
- **Python**: ML pipeline implementation
- **scikit-learn**: Machine learning models
- **pandas**: Data manipulation
- **NLTK**: Natural language processing

## Model Performance

- **Accuracy**: 65.79%
- **Precision**: 73.51%
- **Recall**: 65.79%

---

**Note**: This README serves as a revision guide for the DVC pipeline setup process. 
