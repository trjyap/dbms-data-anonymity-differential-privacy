import pandas as pd
import numpy as np
import anonypy

# Column names from the dataset
columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

# Load dataset from UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
df = pd.read_csv(url, names=columns, na_values="?", skipinitialspace=True)

# Drop rows with missing values (for simplicity)
df.dropna(inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)

print(f"Dataset loaded with {len(df)} records.")
print("First few rows:")
print(df.head())

# --- Define Quasi-Identifiers ---
# These are attributes that could be linked with external data to re-identify individuals
quasi_identifiers = [
    'age',
    'education',  # or 'education-num'
    'marital-status',
    'race',
    'sex',
    'native-country',
    'workclass',
    'occupation'
]
# Note: 'fnlwgt' (final weight) is survey-specific and may be excluded or included based on use case

# We'll keep only quasi-identifiers + income (target) for analysis
df_anon = df[quasi_identifiers].copy()

# Convert all quasi-identifiers to strings for grouping
for col in quasi_identifiers:
    df_anon[col] = df_anon[col].astype(str)

# Compute k-anonymity map (group sizes) using Pandas
grouped = df_anon.groupby(quasi_identifiers).size().reset_index(name='k')
k_min = grouped['k'].min()
total_groups = len(grouped)
print(f"\n🔍 k-Anonymity Analysis Results")
print(f"Total number of equivalence classes: {total_groups}")
print(f"Minimum k value (k-anonymity): {k_min}")

# Optional: Set desired k threshold
desired_k = 5
at_risk_groups = grouped[grouped['k'] < desired_k]

print(f"Number of equivalence classes with k < {desired_k}: {len(at_risk_groups)}")

if len(at_risk_groups) > 0:
    print(f"\n⚠️  Examples of low-k equivalence classes (k < {desired_k}):")
    print(at_risk_groups.head(10))

    # Find original records in vulnerable groups
    # Ensure types match for merge
    for col in quasi_identifiers:
        df[col] = df[col].astype(str)

    vulnerable_records = pd.merge(df, at_risk_groups, on=quasi_identifiers, how='inner')
    print(f"\n👉 {len(vulnerable_records)} individual records belong to high-risk groups.")
else:
    print(f"\n✅ All groups satisfy k ≥ {desired_k}. The dataset is {desired_k}-anonymous in quasi-identifiers.")

# --- Optional: Improve k via generalization (example: age binning) ---
print("\n🔧 Example: Improving k by generalizing 'age' into bins")

df_anon_generalized = df_anon.copy()
df_anon_generalized['age'] = pd.cut(
    df_anon_generalized['age'].astype(int),
    bins=[0, 25, 40, 60, 100],
    labels=['0-25', '26-40', '41-60', '60+']
)

grouped_gen = df_anon_generalized.groupby(quasi_identifiers, observed=True).size().reset_index(name='k')
grouped_gen = grouped_gen[grouped_gen['k'] > 0]
k_min_gen = grouped_gen['k'].min()

print(f"After age generalization, minimum k = {k_min_gen}")
print(f"Number of small groups (k < {desired_k}): {len(grouped_gen[grouped_gen['k'] < desired_k])}")