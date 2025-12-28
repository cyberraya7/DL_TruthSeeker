import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
import kagglehub
import os

print("--- Starting Data Preprocessing ---")
start_time = time.time()

path = "truthseeker/"

# Download dataset from Kaggle (downloads to default cache location)
# kagglehub.dataset_download("chethuhn/network-intrusion-dataset", path=path)
# curl -L -o ~/DL4IDS/data/CICIDS2017/csv/network-intrusion-dataset.zip  https://www.kaggle.com/api/v1/datasets/download/chethuhn/network-intrusion-dataset

#print("Downloaded to cache:", path)

df = pd.DataFrame()

for file in os.listdir(path):
    if file.endswith(".csv"):
        df_temp = pd.read_csv(os.path.join(path, file))
        df = pd.concat([df, df_temp])
        print(f"Successfully loaded {df.shape[0]} rows from {file}.")
    else:
        print(f"No CSV files found in {path}.")



# Check for NaN values
nan_counts = df.isna().sum()
problematic_nan_cols = nan_counts[nan_counts > 0]

if not problematic_nan_cols.empty:
    print("\nFound columns with NaN values:")
    print(problematic_nan_cols)
    print(f"Total NaN values: {nan_counts.sum()}")
    
    # Fill NaN values with the mean of each column (for numeric columns)
    # For non-numeric columns, fill with 0 or the most frequent value
    for col in problematic_nan_cols.index:
        
        # For numeric columns, fill with mean
        mean_val = df[col].mean()
        if pd.isna(mean_val):
            # If mean is also NaN (all values are NaN), use 0
            df[col].fillna(0, inplace=True)
            print(f"  Column '{col}': Filled {nan_counts[col]} NaN values with 0 (mean was also NaN)")
        else:
            df[col].fillna(mean_val, inplace=True)
            print(f"  Column '{col}': Filled {nan_counts[col]} NaN values with mean ({mean_val:.4f})")
        
    
    # Verify NaN are fixed
    remaining_nan = df.isna().sum().sum()
    if remaining_nan > 0:
        print(f"WARNING: {remaining_nan} NaN values still remain after filling!")
    else:
        print("All NaN values have been filled.")
else:
    print("No NaN values found in the dataset.")

# Check for infinity values
inf_counts = df.isin([np.inf, -np.inf]).sum()
problematic_inf_cols = inf_counts[inf_counts > 0]

if not problematic_inf_cols.empty:
    print("\nFound columns with infinity values:")
    print(problematic_inf_cols)
    
    # Replace all infinity values (-inf and inf) with 0. You can also use df.mean() for a different strategy.
    df.replace([np.inf, -np.inf], 0, inplace=True)
    print("All infinity values have been replaced with 0.")
else:
    print("No infinity values found in the dataset.")



# Separate features and labels.
# Assumption: the last column is the label, all others are features.
label_col = df.columns[-1]
y = df[label_col]

# Drop the label from the feature set
feature_df = df.drop(columns=[label_col])

# Keep only numeric feature columns to avoid strings like names, timestamps, etc.
numeric_feature_df = feature_df.select_dtypes(include=[np.number])
dropped_feature_cols = [c for c in feature_df.columns if c not in numeric_feature_df.columns]

if dropped_feature_cols:
    print("\nDropping non-numeric feature columns (not used for training):")
    for col in dropped_feature_cols:
        print(f"  - {col}")

X = numeric_feature_df.values  # use strictly numeric features

# Extract unique class names from the dataset
class_names = sorted(y.unique().tolist())
print(f"\nFound {len(class_names)} unique classes in the dataset:")
for i, class_name in enumerate(class_names):
    print(f"  {i}: {class_name}")

# Convert text labels (e.g., 'Benign') into integers (0, 1, 2...).
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_
print(f"Labels encoded. Found {len(class_names)} classes.")

# Use stratify=y_encoded to ensure class distribution is similar in train and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print("Data split into training and testing sets (80/20 split).")

X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
)
print("Data split into validation and testing sets (50/50 split).")



y_val = y_val.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

def count_class(partition, name):
    _, num_ins = np.unique(partition, return_counts=True)
    for value, count in zip(class_names, num_ins):
        print(f"{name}: Class {value} : {count} times")

#count_class(y_val,"val")
#count_class(y_train,"train")
#count_class(y_test,"test")

train = np.concatenate((X_train, y_train), axis=1)
test = np.concatenate((X_test, y_test), axis=1)
val = np.concatenate((X_val, y_val), axis=1)

# Final verification: Check for any remaining NaN or Inf values
print("\n--- Final Data Verification ---")
for name, array in [('train', train), ('test', test), ('val', val)]:
    # Ensure we are working with a numeric array for NaN/Inf checks.
    # Some datasets may contain non-numeric columns that end up with dtype=object,
    # which would cause np.isnan / np.isinf to fail.
    try:
        numeric_array = array.astype(np.float32)
    except (TypeError, ValueError):
        print(f"{name}: Skipping NaN/Inf check (non-numeric data type: {array.dtype})")
        continue

    nan_count = np.isnan(numeric_array).sum()
    inf_count = np.isinf(numeric_array).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"WARNING: {name} contains {nan_count} NaN and {inf_count} Inf values!")
        numeric_array = np.nan_to_num(numeric_array, nan=0.0, posinf=0.0, neginf=0.0)

    # After checks/fixes, always overwrite the original split with the
    # numeric, cleaned version so it can be safely saved/loaded.
    if name == 'train':
        train = numeric_array
    elif name == 'test':
        test = numeric_array
    else:
        val = numeric_array

    if nan_count == 0 and inf_count == 0:
        print(f"{name}: No NaN or Inf values ✓")
    else:
        print(f"{name}: NaN/Inf values have been replaced with 0 ✓")

# Ensure everything is numeric before saving, to avoid object-dtype .npy files
train = train.astype(np.float32)
test = test.astype(np.float32)
val = val.astype(np.float32)

np.save("train.npy", train)
np.save("test.npy", test)
np.save("val.npy", val)

np.save("class_names.npy", class_names) # Save class names for the final report

for name, array in [('train', train), ('test', test), ('val', val)]: print(f"{name} shape: {array.shape}")

end_time = time.time()
print("\n--- Preprocessing Complete ---")
print("Saved 3 files")
print(f"Total preprocessing time: {(end_time - start_time):.2f} seconds.")