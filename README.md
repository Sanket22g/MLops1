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

# Commit DVC initialization
git add .dvc .dvcignore
git commit -m "Initialize DVC"
git push
```

### Step 6: AWS S3 Setup for DVC Remote Storage

#### 6.1: Install AWS CLI and Required Dependencies

```bash
# Install DVC with S3 support
pip install dvc[s3]

# Install AWS CLI
pip install awscli

# If you encounter dependency conflicts, upgrade packages:
pip install --upgrade boto3 botocore aiobotocore
```

**Note**: Common dependency conflicts:
- `aiobotocore` requires older `botocore` but `awscli` and `boto3` need newer versions
- Solution: `pip install --upgrade aiobotocore>=2.15.0` to get compatible versions

#### 6.2: Configure AWS Credentials

```bash
# Configure AWS credentials
aws configure
```

Provide the following:
- **AWS Access Key ID**: Your AWS access key
- **AWS Secret Access Key**: Your AWS secret key
- **Default region name**: e.g., `ap-southeast-1`
- **Default output format**: Press Enter (leave as None)

Your credentials will be saved in `~/.aws/credentials`

#### 6.3: Create S3 Bucket (via AWS Console or CLI)

**Option 1: AWS Console**
1. Go to AWS S3 Console
2. Click "Create bucket"
3. Name your bucket (e.g., `dvc-mlops-proj1`)
4. Choose region (same as configured)
5. Keep default settings and create

**Option 2: AWS CLI**
```bash
# Create S3 bucket
aws s3 mb s3://dvc-mlops-proj1 --region ap-southeast-1
```

#### 6.4: Configure DVC Remote Storage

```bash
# Add S3 bucket as DVC remote storage
dvc remote add -d dvcstore s3://dvc-mlops-proj1
```

**Note**: If the above command doesn't update `.dvc/config`, manually edit it:

```ini
[core]
    remote = dvcstore
[remote "dvcstore"]
    url = s3://dvc-mlops-proj1
```

#### 6.5: Verify Configuration

```bash
# Check DVC config
cat .dvc/config

# Test AWS connection
aws s3 ls s3://dvc-mlops-proj1
```

#### 6.6: Commit DVC Remote Configuration

```bash
git add .dvc/config
git commit -m "Add S3 remote storage configuration"
git push
```

### Step 7: Create DVC Pipeline (dvc.yaml)

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

### Step 8: Run DVC Pipeline

```bash
# Test pipeline automation
dvc repro

# Check pipeline visualization
dvc dag
```

### Step 9: Commit Pipeline Configuration

```bash
git add dvc.yaml dvc.lock
git commit -m "Add DVC pipeline configuration"
git push
```

### Step 10: Push Data and Models to S3

```bash
# After running dvc repro, push outputs to S3
dvc push

# This will push 7 tracked files:
# 1. raw_data/train_data.csv
# 2. raw_data/test_data.csv
# 3. cleaned_data/train_data_cleaned.csv
# 4. cleaned_data/test_data_cleaned.csv
# 5. vectorized_data/x_train_tfidf.csv
# 6. vectorized_data/x_test_tfidf.csv
# 7. models/random_forest_model.joblib

# Verify files in S3
aws s3 ls s3://dvc-mlops-proj1 --recursive
```

### Step 11: Pull Data from S3 (On Another Machine)

```bash
# Clone the Git repository
git clone <your-repo-url>
cd MLops1

# Configure AWS credentials (same as Step 6.2)
aws configure

# Pull data from S3
dvc pull

# Now all data and models are restored locally
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

## AWS S3 and DVC Troubleshooting

### Issue 1: "No remote specified" Error

**Problem**: `dvc push` fails with "no remote specified"

**Solution**: 
```bash
# Check if .dvc/config is empty
cat .dvc/config

# If empty, manually add configuration:
# Edit .dvc/config and add:
[core]
    remote = dvcstore
[remote "dvcstore"]
    url = s3://your-bucket-name
```

### Issue 2: Dependency Conflicts (boto3, botocore, aiobotocore)

**Problem**: After installing `dvc[s3]` and `awscli`, you get version conflicts

**Solution**:
```bash
# Upgrade all AWS-related packages to compatible versions
pip install --upgrade boto3>=1.42.44 botocore>=1.42.44 aiobotocore>=2.15.0
```

### Issue 3: AWS Credentials Not Working

**Problem**: DVC can't access S3 bucket

**Solution**:
```bash
# Verify AWS credentials
aws configure list

# Test S3 access
aws s3 ls s3://your-bucket-name

# If it fails, reconfigure:
aws configure
```

### Issue 4: Files Not Pushing to S3

**Problem**: `dvc push` says 0 files pushed

**Solution**:
```bash
# Run pipeline first to generate outputs
dvc repro

# Commit changes
dvc commit

# Then push
dvc push
```

## Technologies Used

- **DVC**: Data Version Control and pipeline automation
- **Git**: Code version control
- **Amazon S3**: Remote data storage
- **Python**: ML pipeline implementation
- **scikit-learn**: Machine learning models
- **pandas**: Data manipulation
- **NLTK**: Natural language processing

## Complete Workflow: Git + DVC

### For Code Changes:
```bash
# 1. Make changes to code files
# 2. Add to Git
git add src/
git add dvc.yaml params.yaml

# 3. Commit
git commit -m "Update feature engineering"

# 4. Push to GitHub
git push origin main
```

### For Data/Model Changes:
```bash
# 1. Run pipeline to generate new data/models
dvc repro

# 2. Commit DVC metadata
dvc commit

# 3. Push data to S3
dvc push

# 4. Commit .dvc/lock file to Git
git add dvc.lock
git commit -m "Update pipeline outputs"
git push origin main
```

### For Team Collaboration:
```bash
# Team member pulls your changes:
git pull origin main    # Get code changes
dvc pull               # Get data/models from S3
```

## What Gets Stored Where?

| Item | Stored In | Command |
|------|-----------|---------|
| Code files (`.py`, `.yaml`) | Git (GitHub) | `git push` |
| Data files (`.csv`) | DVC (S3) | `dvc push` |
| Models (`.joblib`) | DVC (S3) | `dvc push` |
| Pipeline config (`dvc.yaml`) | Git (GitHub) | `git push` |
| DVC metadata (`dvc.lock`) | Git (GitHub) | `git push` |
| Metrics (`.json`) | Git (GitHub) | `git push` |

## Model Performance

- **Accuracy**: 65.79%
- **Precision**: 73.51%
- **Recall**: 65.79%

---

**Note**: This README serves as a revision guide for the DVC pipeline setup process. 
