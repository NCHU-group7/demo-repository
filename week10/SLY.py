import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# Read the data from the file week10.csv
df = pd.read_csv('week10/week10.csv', sep = ';')

# List of features to analyze
features = ["age", "job", "marital", "education", "loan"]

# Create histograms for each feature separated by term deposit "y"
for feature in features:
    yes_counts = df[df['y'] == 'yes'][feature].value_counts().sort_index()
    no_counts = df[df['y'] == 'no'][feature].value_counts().sort_index()
    
    indices = sorted(set(yes_counts.index).union(set(no_counts.index)))
    yes_counts = yes_counts.reindex(indices, fill_value=0)
    no_counts = no_counts.reindex(indices, fill_value=0)
    
    bar_width = 0.35
    index = np.arange(len(indices))
    
    plt.bar(index, yes_counts, bar_width, alpha=0.5, color='blue', label='Yes')
    plt.bar(index + bar_width, no_counts, bar_width, alpha=0.5, color='red', label='No')
    
    plt.xticks(index + bar_width / 2, indices)
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {feature} by Term Deposit')
    plt.legend()
    plt.show()

    # Select features and target variable
    X = df[["age", "balance", "loan"]]
    X['loan'] = X['loan'].apply(lambda x: 1 if x == 'yes' else 0)
    y = df['y'].apply(lambda x: 1 if x == 'yes' else 0)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create and train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate and print the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    