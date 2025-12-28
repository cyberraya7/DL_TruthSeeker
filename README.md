## TruthSeeker-Deep Learning Assignment

## 1. Objectives

The **TruthSeeker** project builds an end‑to‑end deep learning pipeline on the `Truth_Seeker_Model_Dataset` to perform **multi‑class classification** of human judgment outcomes.  
Each sample is mapped into one of five agreement classes: **Agree**, **Mostly Agree**, **NO MAJORITY**, **Disagree**, and **Mostly Disagree**, using tabular features.  
The objective is to compare a strong baseline **MLP** against a more expressive **Transformer‑based** architecture for this tabular stance/consensus prediction task and to prepare the best model for deployment (ONNX export).

## 2. Key Phases

- **Data Selection & Loading**  
  - Used the `truthseeker/Truth_Seeker_Model_Dataset.csv` file(s) under `data_process/truthseeker/`.  
  - All CSV files in this folder are concatenated into a single Pandas DataFrame.

- **Data Cleaning & Preprocessing (Offline)**  
  - Handled **NaN** values by filling numeric columns with their mean (or 0 if the mean is undefined).  
  - Replaced all occurrences of **±∞** with 0.  
  - Selected **only numeric feature columns**, dropping non‑numeric fields such as IDs or timestamps.  
  - Encoded the textual label column into integer class indices with `LabelEncoder`, and saved the corresponding `class_names` (also as `class_names.npy`).

- **Data Splitting**  
  - Performed an **80/20 train–test split** with stratification on the encoded labels to preserve class distribution.  
  - Further split the temporary test set **50/50 into validation and final test** (resulting in train/val/test ≈ 80/10/10), again stratified by labels.  
  - Combined features and labels into NumPy arrays and saved them as `train.npy`, `val.npy`, and `test.npy`.

- **Model‑Ready Preprocessing (Online)**  
  - In `luq_preprocess.py`, fitted a **StandardScaler** on the training features and applied it to train/val/test.  
  - Converted arrays into `TensorDataset`s and wrapped them with `DataLoader`s (shuffled training, non‑shuffled validation and test).  
  - Optionally persisted the fitted scaler to `scaler.pkl` for later ONNX / deployment inference.

- **Experiment Tracking**  
  - Used **Weights & Biases (wandb)** to log training/validation losses, learning rate schedules, and final accuracies for both MLP and Transformer runs.

## 3. Model Architecture

Both models are **multi‑class classifiers** trained with cross‑entropy loss on standardized tabular features, predicting one of the five agreement classes.

### 3a. MLP

- **File**: `model.py` (used by `luq_truthseeker.py`)  
- **Input**: Flattened numeric feature vector of size `input_features = train.shape[1] - 1` (all columns except the label).  
- **Architecture**:
  - Fully connected layer: `input_features → 256`, followed by **BatchNorm1d**, **GELU**, and **Dropout(p=0.2)**.  
  - Fully connected layer: `256 → 128`, again with **BatchNorm1d**, **GELU**, and **Dropout(p=0.2)**.  
  - Output layer: `128 → num_classes` (5), producing logits for the agreement classes.  
- **Motivation**:  
  - A deep **MLP** is a natural baseline for tabular data, capturing non‑linear interactions between standardized numeric features.  
  - Batch normalization stabilizes training; GELU activations and dropout help with expressiveness and regularization.

### 3b. Transformer

- **File**: `luq_transformer.py` (used by `luq_transformer_model.py`)  
- **Input Interpretation**: Treats each scalar feature as a **token** in a sequence of length `input_features`.  
- **Embedding & Encoder**:
  - Projects each scalar feature from shape `(B, F)` to `(B, F, 1)` and then through a linear layer `1 → d_model` (default `d_model = 64`).  
  - Applies a stack of **TransformerEncoderLayer** modules (`nhead = 4`, `num_layers = 2`, `dim_feedforward = 128`, `dropout = 0.1`, GELU activation, `batch_first=True`).  
  - Adds a final **LayerNorm** over the embedding dimension.  
- **Pooling & Head**:
  - Averages over the feature (sequence) dimension to obtain a single `(B, d_model)` representation.  
  - Passes this through a classifier head: `d_model → 128 → num_classes` with **GELU** and **Dropout** in between.  
- **Motivation**:  
  - Self‑attention allows the model to learn **interactions between features** that may be difficult to capture with an MLP.  
  - This makes the Transformer model a strong candidate for complex tabular patterns in the TruthSeeker dataset.

## 4. Training & Experimentation

- **Scripts**:  
  - `luq_truthseeker.py` (MLP baseline).  
  - `luq_transformer_model.py` (Transformer‑based model).

- **Hyperparameters** (from `config.yaml`):  
  - **Batch size**: 128  
  - **Learning rate**: 0.01 (AdamW)  
  - **Epochs**: 20  
  - **Loss function**: `CrossEntropyLoss` for 5‑class classification.

- **Optimization & Regularization**  
  - Optimizer: **AdamW** for both models.  
  - Scheduler: **ReduceLROnPlateau** on validation loss (factor 0.1, patience 3) to reduce LR when progress stalls.  
  - Gradient clipping (`max_norm = 1.0`) to stabilize training.  
  - Early stopping behavior via tracking **best validation loss** with a patience of 5 epochs; best model weights are saved to `.pth` (`luq_truthseeker.pth` or `luq_transformer.pth`).

- **Experiment Tracking (wandb)**  
  - Logged per‑epoch **train loss**, **validation loss**, and **learning rate**.  
  - Logged final **test accuracy** for each model.  
  - Maintained separate wandb projects for the MLP (`TruthSeeker`) and Transformer (`Transformer`) runs, enabling side‑by‑side comparison of learning curves and performance.

## 5. Evaluation

- **Held‑Out Test Set**  
  - Final models are evaluated on the **test loader** built from `test.npy`, which was never used during training or validation.  
  - Both scripts (`luq_truthseeker.py` and `luq_transformer_model.py`) load the best checkpoint (based on validation loss) before inference on the test set.

- **Metrics**  
  - **Accuracy**: Overall proportion of correctly classified samples, printed as `Final Test Accuracy` and logged to wandb.  
  - **Precision, Recall, F1‑Score**: Computed per class using `sklearn.metrics.classification_report`, with care taken to align class indices with `class_names`.  
  - **Confusion Matrix**: Computed using `sklearn.metrics.confusion_matrix` to visualize misclassifications across the five agreement classes.

- **Efficiency & Deployment**  
  - Training is performed on **GPU if available**, otherwise CPU, and gradient clipping plus learning‑rate scheduling help maintain stable and efficient optimization.  
  - The best MLP model is exported to **ONNX** (`luq_truthseeker.onnx`) via `luq_convert.py`, with dynamic batch axes, enabling optimized deployment and fast inference (e.g., with ONNX Runtime).  
  - The same preprocessing pipeline (saved `StandardScaler` and DataLoader construction) is designed to be reused at inference time to keep training and deployment distributions aligned.


