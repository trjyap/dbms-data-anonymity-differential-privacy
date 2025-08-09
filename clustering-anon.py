import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

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

# Reset index
df.reset_index(drop=True, inplace=True)

print(f"Dataset loaded with {len(df)} records.")
print("First few rows:")
print(df.head())

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
print(df['cluster'].value_counts())