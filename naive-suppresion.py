import pandas as pd

# Load the data
columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
df = pd.read_csv(url, names=columns, sep=', ', engine='python', na_values='?')

# Drop rows with missing values
df.dropna(inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)

print(f"Dataset loaded with {len(df)} records.")
print("First few rows:")
print(df.head())

quasi_identifiers = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'race', 'sex', 'native-country', 'capital-gain', 'capital-loss']

k = 5

# Count frequency of each QI combination
qi_groups = df[quasi_identifiers].groupby(quasi_identifiers).size().reset_index(name='count')

# Filter: keep only combinations that appear at least k times
frequent_qi_combinations = qi_groups[qi_groups['count'] >= k]

# Merge back to keep only records with frequent combinations
df_anonymized = df.merge(frequent_qi_combinations[quasi_identifiers], on=quasi_identifiers, how='inner')

post_suppression_counts = df_anonymized[quasi_identifiers].groupby(quasi_identifiers).size()
print("Minimum group size (k):", post_suppression_counts.min())
print("Maximum group size:", post_suppression_counts.max())
print("Number of unique QI groups:", len(post_suppression_counts))

original_size = len(df)
anonymized_size = len(df_anonymized)
suppression_rate = (original_size - anonymized_size) / original_size

print(f"Original records: {original_size}")
print(f"Anonymized records: {anonymized_size}")
print(f"Suppression rate: {suppression_rate:.2%}")

# ==============================================================================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Function to preprocess and train
def train_model(data, name):
    df_model = data.copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in df_model.select_dtypes(include='object').columns:
        df_model[col] = le.fit_transform(df_model[col])
    
    X = df_model.drop('income', axis=1)
    y = df_model['income']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} - Accuracy: {acc:.4f}")
    return acc

# Train on original and anonymized
acc_original = train_model(df, "Original Dataset")
acc_anonymized = train_model(df_anonymized, "Naively Suppressed Dataset")

print(f"Accuracy drop: {acc_original - acc_anonymized:.4f}")

import matplotlib.pyplot as plt

# Income distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

df['income'].value_counts().plot(kind='bar', ax=axes[0], title='Original: Income Distribution')
df_anonymized['income'].value_counts().plot(kind='bar', ax=axes[1], title='After Suppression: Income Distribution')

plt.tight_layout()
plt.show()