import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = 'C:/Users/Liang/demo-repository/Proposal/wine/winequality-white.csv'
data = pd.read_csv(file_path, sep=';')

# Step 1: Handle Missing Values
# Check for missing values
missing_values = data.isnull().sum()
print("Missing values per column:\n", missing_values)

# If missing values exist, impute or drop them
data = data.dropna()  # Example: drop rows with missing values

# Step 2: Remove Duplicates
data = data.drop_duplicates()

# Step 3: Standardize Numerical Features
# Identify numerical columns (excluding the target 'quality')
numerical_columns = data.columns.drop('quality')
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Step 4: Split the Dataset into Training and Test Sets
X = data.drop('quality', axis=1)
y = data['quality']

# Create train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 5: Save the Processed Data
# Save the train and test sets as CSV files for later use
X_train.to_csv('Proposal/X_train.csv', index=False)
X_test.to_csv('Proposal/X_test.csv', index=False)
y_train.to_csv('Proposal/y_train.csv', index=False)
y_test.to_csv('Proposal/y_test.csv', index=False)

print("Preprocessing complete. Train and test sets saved.")
