import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from collections import Counter

# Load Adult dataset
columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
df = pd.read_csv(url, names=columns, sep=', ', engine='python', na_values='?')

# Drop missing values
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Keep only quasi-identifiers and sensitive attribute
quasi_identifiers = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'race', 'sex', 'native-country', 'capital-gain', 'capital-loss']
sensitive = 'income'

df_qi = df[quasi_identifiers + [sensitive]].copy()

# Ensure numeric types before discretization
for col in ['age', 'capital-gain', 'capital-loss']:
    df_qi[col] = pd.to_numeric(df_qi[col], errors='coerce')

# Discretize age
df_qi['age'] = pd.cut(df_qi['age'], bins=5, labels=False)  # 5 age groups

# Optional: cap extreme capital gains/losses
for col in ['capital-gain', 'capital-loss']:
    df_qi[col] = pd.cut(df_qi[col], bins=3, labels=False)

def get_distribution(series):
    """Return normalized distribution of values in a series."""
    c = Counter(series)
    total = sum(c.values())
    return np.array([c[v] / total for v in sorted(c.keys())])

def t_closeness_check(df, qi_cols, sensitive_col, k=5, t=0.2):
    """
    Apply k-anonymity + t-closeness via suppression.
    Returns anonymized DataFrame.
    """
    # Step 1: Apply k-anonymity (suppression)
    freq = df[qi_cols].value_counts()
    keep_idx = df[qi_cols].apply(tuple, axis=1).map(freq) >= k
    df_kanon = df[keep_idx].copy()

    if len(df_kanon) == 0:
        return df_kanon

    # Step 2: Get global distribution of sensitive attribute
    unique_vals = sorted(df_kanon[sensitive_col].unique())
    label_to_pos = {val: idx for idx, val in enumerate(unique_vals)}
    
    def get_probs(series):
        dist = series.value_counts(normalize=True).reindex(unique_vals, fill_value=0)
        return dist.values, np.array([label_to_pos[v] for v in unique_vals])
    
    global_probs, positions = get_probs(df_kanon[sensitive_col])

    # Step 3: Group by QIs and filter t-closeness violations
    valid_groups = []

    for _, group in df_kanon.groupby(qi_cols):
        if len(group) == 0:
            continue
        group_probs, _ = get_probs(group[sensitive_col])
        
        # Compute Wasserstein distance (1D EMD)
        try:
            emd_dist = wasserstein_distance(positions, positions, global_probs, group_probs)
        except:
            emd_dist = float('inf')
        
        if emd_dist <= t:
            valid_groups.append(group)

    # Return combined valid groups
    return pd.concat(valid_groups, ignore_index=True) if valid_groups else pd.DataFrame(columns=df.columns)

def apply_k_anonymity(df, qi_cols, k=5):
    """Apply suppression to enforce k-anonymity."""
    counts = df[qi_cols].groupby(qi_cols).size()
    frequent_groups = counts[counts >= k].index
    return df.set_index(qi_cols).index.isin(frequent_groups), counts

# Get mask of records in frequent groups
mask, group_counts = apply_k_anonymity(df_qi, quasi_identifiers, k=5)
df_kanon = df_qi[mask].copy()

global_dist = get_distribution(df_kanon[sensitive])
print("Global income distribution:", global_dist)
# Example: [0.75, 0.25] → 75% <=50K, 25% >50K


t = 0.2  # t-closeness threshold
valid_groups = []

# Group by all QI columns
grouped = df_kanon.groupby(quasi_identifiers)

for name, group in grouped:
    result = t_closeness_check(group, quasi_identifiers, sensitive, k=5, t=t)
    if not result.empty:
        valid_groups.append(result)

# Combine valid groups
if valid_groups:
    df_tcloseness = pd.concat(valid_groups, ignore_index=True)
else:
    df_tcloseness = pd.DataFrame(columns=df_kanon.columns)

print(f"t-closeness applied. t={t}")
print(f"k-anonymity + t-closeness dataset size: {len(df_tcloseness)}")
print(f"Suppression rate: {(len(df_qi) - len(df_tcloseness)) / len(df_qi):.2%}")

# =========================================== Privacy Evaluation ===========================================

print("\n=== Privacy Evaluation ===")
print(f"Final dataset size: {len(df_tcloseness)}")
print(f"k-anonymity satisfied: {df_tcloseness[quasi_identifiers].value_counts().min() >= 5}")
print(f"t-closeness threshold (t): {t}")

# Check t-closeness violations
violation_count = 0
grouped = df_tcloseness.groupby(quasi_identifiers)

for name, group in grouped:
    result = t_closeness_check(group, quasi_identifiers, sensitive, k=5, t=t)
    if result.empty:
        violation_count += 1

print(f"Groups violating t-closeness: {violation_count}")


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def evaluate_utility(df_original, df_anon, target='income'):
    le = LabelEncoder()
    X_orig = df_original.drop(target, axis=1).copy()
    for col in X_orig.select_dtypes(include='object').columns:
        X_orig[col] = le.fit_transform(X_orig[col].astype(str))
    
    y_orig = le.fit_transform(df_original[target])
    
    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, test_size=0.3, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    return accuracy_score(y_test, y_pred)

acc_original = evaluate_utility(df_qi, df_qi)
acc_tcloseness = evaluate_utility(df_tcloseness, df_tcloseness)

# ============================================ Utility Evaluation ===========================================

print(f"\n=== Utility Evaluation ===")
print(f"Original accuracy: {acc_original:.4f}")
print(f"t-closeness accuracy: {acc_tcloseness:.4f}")
print(f"Accuracy drop: {acc_original - acc_tcloseness:.4f}")

print(f"Final dataset size after k-anonymity + t-closeness: {len(df_tcloseness)}")