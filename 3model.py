import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

# 1. Memuat Data
df = pd.read_csv('water_potability.csv')

# 2. Analisis Data
print("\n# 2. Analisis Data")
print("Informasi Dataset:")
print(df.info())
print("\nNilai Unik:")
print(df.nunique())

# 3. Validasi Data dan Visualisasi
print("\n# 3. Validasi Data")
print("Data yang Hilang:")
print(df.isnull().sum())

# Menangani nilai yang hilang dengan imputasi rata-rata
df = df.fillna(df.mean())

# Membuat plot distribusi target
plt.figure(figsize=(10, 5))
df['Potability'].value_counts().plot(kind='bar')
plt.title('Distribution of Water Potability')  # Judul dalam Bahasa Inggris
plt.xlabel('Potability')
plt.ylabel('Count')
plt.savefig('distribution.png')
plt.close()

# 4. Menentukan Fitur dan Target
X = df.drop('Potability', axis=1)
y = df['Potability']

print("\n# 4. Fitur dan Target")
print("Fitur:", X.columns.tolist())
print("Target: Potability")

# 5. Pembersihan Data dan Visualisasi
# Heatmap Korelasi
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Heatmap')  # Judul dalam Bahasa Inggris
plt.savefig('correlation_heatmap.png')
plt.close()

# Plot distribusi fitur
plt.figure(figsize=(15, 10))
for i, col in enumerate(X.columns, 1):
    plt.subplot(3, 3, i)
    plt.hist(X[col], bins=30)
    plt.title(col)  
plt.tight_layout()
plt.savefig('feature_distributions.png')
plt.close()

# 6. Konstruksi Data
print("\n# 6. Tipe Data:")
print(df.dtypes)
# 7. Pemodelan
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Normalisasi fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Inisialisasi model
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Regresi Logistik': LogisticRegression(random_state=42),
    'SVM': SVC(random_state=42)
}

# Melatih dan mengevaluasi model
print("\n# 7. Hasil Model:")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\n{name}:")
    print(f"Akurasi: {accuracy:.4f}")
    print("Matriks Konfusi:")
    print(conf_matrix)

# 8. Evaluasi Model
print("\n# 8. Evaluasi Model")
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} - Akurasi Akhir: {accuracy:.4f}")