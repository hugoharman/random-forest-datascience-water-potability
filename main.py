import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

# 1. Memuat Data
df = pd.read_csv('water_potability.csv')

# 2. Analisis Data
print("\n# 2. Analisis Data")
print("Informasi Dataset:")
print(df.info())
print("\nNilai Unik:")
print(df.nunique())
print("\nStatistik Deskriptif:")
print(df.describe())

# 3. Validasi Data
print("\n# 3. Validasi Data")
print("Data yang Hilang:")
print(df.isnull().sum())

# 4. Pembersihan Data
def handle_outliers(df, columns):
    df_clean = df.copy()
    for column in columns:
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean[column] = df_clean[column].clip(lower_bound, upper_bound)
    return df_clean

# Copy dataframe
df_cleaned = df.copy()

# Handle missing values
for column in df_cleaned.columns:
    if df_cleaned[column].isnull().sum() > 0:
        if df_cleaned[column].dtype in ['int64', 'float64']:
            df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].median())

# Handle outliers
df_cleaned = handle_outliers(df_cleaned, df_cleaned.drop('Potability', axis=1).columns)

# Handle skewness
for column in df_cleaned.columns:
    if column != 'Potability':
        skewness = df_cleaned[column].skew()
        if abs(skewness) > 1:
            df_cleaned[column] = np.log1p(df_cleaned[column] - df_cleaned[column].min() + 1)

# Target distribution plot
plt.figure(figsize=(10, 5))
df_cleaned['Potability'].value_counts().plot(kind='bar')
plt.title('Distribution of Water Potability')
plt.xlabel('Potability')
plt.ylabel('Count')
plt.savefig('distribution.png')
plt.close()

# 5. Features and Target
X = df_cleaned.drop('Potability', axis=1)
y = df_cleaned['Potability']

print("\n# 5. Fitur dan Target")
print("Fitur:", X.columns.tolist())
print("Target: Potability")

# 6. Visualizations
plt.figure(figsize=(12, 10))
sns.heatmap(df_cleaned.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()

plt.figure(figsize=(15, 10))
for i, col in enumerate(X.columns, 1):
    plt.subplot(3, 3, i)
    plt.hist(X[col], bins=30)
    plt.title(f'{col} Distribution')
plt.tight_layout()
plt.savefig('feature_distributions.png')
plt.close()

# 7. Modeling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.close()

print("\nRandom Forest:")
print(f"Akurasi: {accuracy:.4f}")
print("Matriks Konfusi:")
print(conf_matrix)

# 8. Save Model and Scaler
joblib.dump(model, 'random_forest_model.pkl')
joblib.dump(scaler, 'standard_scaler.pkl')
print("\nModel dan Scaler berhasil disimpan!")