import pandas as pd
import time
import psutil
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Column names from the dataset
columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

# Load dataset from UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
df = pd.read_csv(url, names=columns, na_values="?", skipinitialspace=True)

# Drop rows with missing values
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"Dataset loaded with {len(df)} records.")
print("First few rows:")
print(df.head())

# --- Runtime (seconds) ---
start_time = time.time()

# Select quasi-identifiers
quasi_identifiers = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'race', 'sex', 'native-country', 'capital-gain', 'capital-loss']

# Encode categorical variables
df_encoded = df[quasi_identifiers].copy()
for col in df_encoded.select_dtypes(include='object'):
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(df_encoded)

# Apply clustering
k_clusters = 10  # Choose based on your needs
kmeans = KMeans(n_clusters=k_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Generalize within clusters (example: show age range per cluster)
generalized = df.groupby('cluster')[quasi_identifiers].agg(lambda x: f"{x.min()}-{x.max()}")
print(generalized)

# Check cluster sizes (for k-anonymity)
cluster_sizes = df['cluster'].value_counts()
print(cluster_sizes)

# --- Metrics ---

k = 5

# 1. k-anonymity satisfied? (min group size >= k)
k_anonymity_satisfied = cluster_sizes.min() >= k
print(f"k-anonymity satisfied? {k_anonymity_satisfied}")

# 2. Uniqueness (% of clusters with only one record)
unique_clusters = cluster_sizes[cluster_sizes == 1]
uniqueness_rate = len(unique_clusters) / k_clusters if k_clusters > 0 else 0
print(f"Uniqueness rate: {uniqueness_rate:.2%}")

# 3. Re-identification risk estimate (simulated risk)
risk_per_cluster = 1 / cluster_sizes
risk_per_record = df['cluster'].map(risk_per_cluster)
reid_risk = risk_per_record.mean()
print(f"Estimated re-identification risk: {reid_risk:.4f}")

# 4. Suppression Rate (% of records removed)
# If you want to suppress clusters smaller than k:
suppressed_records = df[df['cluster'].map(cluster_sizes) < k]
suppression_rate = len(suppressed_records) / len(df)
print(f"Suppression rate: {suppression_rate:.2%}")

# 5. Information Loss (Normalized Certainty Penalty, NCP)
ncp_total = 0
for col in quasi_identifiers:
    orig_unique = df[col].nunique()
    # For generalized, count unique ranges
    anon_unique = generalized[col].nunique()
    ncp_col = (orig_unique - anon_unique) / orig_unique if orig_unique > 0 else 0
    ncp_total += ncp_col
ncp_avg = ncp_total / len(quasi_identifiers)
print(f"Information Loss (NCP): {ncp_avg:.4f}")

# --- Model Accuracy ---
def train_model(data, name):
    df_model = data.copy()
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

acc_original = train_model(df, "Original Dataset")
# If you want to train on only non-suppressed records:
acc_anonymized = train_model(df[df['cluster'].map(cluster_sizes) >= k], "Clustered (k-anonymous) Dataset")
print(f"Accuracy drop: {acc_original - acc_anonymized:.4f}")

# --- Memory Usage (Approximate peak RAM) ---
process = psutil.Process()
mem_usage_mb = process.memory_info().rss / (1024 * 1024)
print(f"Approximate peak RAM usage: {mem_usage_mb:.2f} MB")

# --- Runtime (seconds) ---
runtime_seconds = time.time() - start_time
print(f"Runtime (seconds): {runtime_seconds:.2f}")