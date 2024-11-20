import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv('week11 handout/train.csv')
print(df.head(10))

# Handle missing values
df.fillna(df.select_dtypes(include=[np.number]).mean(), inplace=True)
# Select relevant columns
df = df[['Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Pclass', 'Survived']]
# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('Survived', axis=1))

# Convert the scaled data back to a DataFrame
scaled_df = pd.DataFrame(scaled_features, columns=df.drop('Survived', axis=1).columns)
scaled_df['Survived'] = df['Survived'].values

print(scaled_df.head(10))
# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_df.drop('Survived', axis=1))
principal_df = pd.DataFrame(data=principal_components, columns=['principal_component_1', 'principal_component_2'])

# Concatenate the target variable
final_df = pd.concat([principal_df, df[['Survived']]], axis=1)

print(final_df.head(10))

# Split the data into training and testing sets
X = scaled_df.drop('Survived', axis=1)
y = scaled_df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print("Precision:", precision_score(y_test, y_pred_log_reg, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_log_reg, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred_log_reg, average='weighted'))


# Decision Tree
dec_tree = DecisionTreeClassifier()
dec_tree.fit(X_train, y_train)
y_pred_dec_tree = dec_tree.predict(X_test)

print("\nDecision Tree:")
print("Accuracy:", accuracy_score(y_test, y_pred_dec_tree))
print("Precision:", precision_score(y_test, y_pred_dec_tree, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_dec_tree, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred_dec_tree, average='weighted'))


# Plot confusion matrix for Logistic Regression
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)
plt.imshow(conf_matrix_log_reg, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Logistic Regression Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(np.unique(y)))
plt.xticks(tick_marks, np.unique(y))
plt.yticks(tick_marks, np.unique(y))
plt.xlabel('Predicted label')
plt.ylabel('True label')

# Show numbers in blocks
for i in range(conf_matrix_log_reg.shape[0]):
    for j in range(conf_matrix_log_reg.shape[1]):
        plt.text(j, i, format(conf_matrix_log_reg[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix_log_reg[i, j] > conf_matrix_log_reg.max() / 2. else "black")

# Plot confusion matrix for Decision Tree
plt.subplot(1, 2, 2)
conf_matrix_dec_tree = confusion_matrix(y_test, y_pred_dec_tree)
plt.imshow(conf_matrix_dec_tree, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Decision Tree Confusion Matrix')
plt.colorbar()
plt.xticks(tick_marks, np.unique(y))
plt.yticks(tick_marks, np.unique(y))
plt.xlabel('Predicted label')
plt.ylabel('True label')

# Show numbers in blocks
for i in range(conf_matrix_dec_tree.shape[0]):
    for j in range(conf_matrix_dec_tree.shape[1]):
        plt.text(j, i, format(conf_matrix_dec_tree[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix_dec_tree[i, j] > conf_matrix_dec_tree.max() / 2. else "black")

plt.tight_layout()
plt.show()