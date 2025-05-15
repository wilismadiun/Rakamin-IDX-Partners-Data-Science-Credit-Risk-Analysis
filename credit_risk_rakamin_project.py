# %% [markdown]
# # Credit Risk Classification

# %% [markdown]
# # Data Understanding

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %% [markdown]
# ### Data Loading

# %%
df = pd.read_csv("loan_data_2007_2014.csv")

df.head()

# %% [markdown]
# ### Identifikasi Struktur Dataset

# %%
df.shape

# %%
df.info()

# %% [markdown]
# Terdapat beberapa kolom/fitur yang jumlah data kosong lebih dari 50%. Hal tersebut dapat menyebabkan bias. Jadi, langkah selanjutnya adalah fitur yang mengalami lebih dari 50% data hilang akan dihapus. 

# %%
missing_value = df.isna().sum() / df.shape[0]

drop_column = []

for i in missing_value.keys():
    if missing_value[i] >= 0.5:
        drop_column.append(i)
    
drop_column

# %%
df = df.drop(columns=drop_column, axis=0)

df.shape

# %% [markdown]
# ### Membuat kolom untuk label

# %%
df["loan_status"].unique()

# %% [markdown]
# Karena terdapat beberapa data yang masih dalam proses pembayaran yang ditujukan dengan nilai **Current**, maka baris tersebuat akan dihapus. Hal tersebut dilakukan karena pinjaman yang sedang berjalan belum bisa dikatakan sebagai gagal bayar atau pembayaran berhasil.

# %%
df = df[df["loan_status"] != "Current"]

df.shape

# %%
# Mengklasifikasikan sebagai GOOD
good_statuses = ['Fully Paid', 'Does not meet the credit policy. Status:Fully Paid']

# Mengklasifikasikan sebagai BAD
bad_statuses = ['Charged Off', 'Default', 'Late (31-120 days)', 'Does not meet the credit policy. Status:Charged Off']

df.loc[df['loan_status'].isin(good_statuses), 'loan_outcome'] = "GOOD"
df.loc[df['loan_status'].isin(bad_statuses), 'loan_outcome'] = "BAD"

df = df.drop('loan_status', axis=1)

df.loan_outcome.value_counts()

# %%
df.shape

# %% [markdown]
# ### Menghapus Kolom yang Tidak Relevan

# %% [markdown]
# Terdapat beberapa kolom yang tidak relevan 
# 
# Terdapat beberapa kolom/fitur yang diinputkan setelah peminjaman disetujui atau dalam tahap pembayaaran. Sedangkan tujuan dari projek ini adalah untuk membuat model yang mampu memprediksi risiko peminjaman. Sehingga fitur yang dianalisa hanya fitur yang didapatkan saat pengajuan peminjaman

# %%
dropped_columns = ['Unnamed: 0', 'id', 'member_id', 'emp_title', 'url', 'zip_code', 'addr_state', 'total_pymnt', 'total_pymnt_inv', 'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d', 'policy_code', 'title']

df = df.drop(columns=dropped_columns, axis=1)

df.shape

# %% [markdown]
# ### Handling Missing Value

# %%
missing_value = df.isnull().sum()

missing_value[missing_value>0]

# %%
# Menghapus baris dengan missing value pada target
df.dropna(subset=['loan_outcome'], inplace=True)

# %%
# Imputasi dan buat fitur indikator missing untuk kolom dengan missing value tinggi
columns_to_impute_high_missing = ['tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim']
for col in columns_to_impute_high_missing:
    if col in df.columns and df[col].isnull().sum() > 0:
        df[col + '_missing_indicator'] = df[col].isnull().astype(int)
        df[col].fillna(df[col].median(), inplace=True) # Imputasi dengan median

# %%
if 'emp_length' in df.columns:
    df['emp_length'].fillna(df['emp_length'].mode()[0], inplace=True)

# %%
# Menangani kolom dengan missing value sedikit 
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

# %%
# Verifikasi missing values setelah penanganan
print("\nJumlah Missing Values Setelah Penanganan:")
print(df.isnull().sum())

# %% [markdown]
# # Exploratory Data Analysis

# %%
kategorical_features = df.select_dtypes(include=['object']).columns
numeric_features = df.select_dtypes(include=['number']).columns

# %% [markdown]
# ## Analisis Univariat

# %% [markdown]
# #### Analisis data numerik

# %%
for col in numeric_features:
    print(f"\n{'='*30} Analisis Univariat Kolom Numerik: {col} {'='*30}")
    print(df[col].describe())
    print(f"Skewness: {df[col].skew():.2f}")
    print(f"Kurtosis: {df[col].kurt():.2f}")
    print(f"Mode: {df[col].mode().iloc[0] if not df[col].mode().empty else None}") # Menampilkan mode pertama jika ada

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribusi {col}')
    plt.xlabel(col)
    plt.ylabel('Frekuensi')

    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col])
    plt.title(f'Box Plot {col}')
    plt.xlabel(col)

    plt.tight_layout()
    plt.show()

# %% [markdown]
# Berdasarkan boxplot di atas, outlier tidak disebabkan oleh kesalahan input (human error). Oleh karena itu, kita tidak disarankan menghapus data outlier, karena dapat menyebabkan hilangnya informasi atau insight yang berharga. Untuk mengatasi hal tersebut digunakanlah standarisasi 

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# %%
for col in numeric_features:
    print(f"\n{'='*30} Analisis Univariat Kolom Numerik: {col} {'='*30}")
    print(df[col].describe())
    print(f"Skewness: {df[col].skew():.2f}")
    print(f"Kurtosis: {df[col].kurt():.2f}")
    print(f"Mode: {df[col].mode().iloc[0] if not df[col].mode().empty else None}") # Menampilkan mode pertama jika ada

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribusi {col}')
    plt.xlabel(col)
    plt.ylabel('Frekuensi')

    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col])
    plt.title(f'Box Plot {col}')
    plt.xlabel(col)

    plt.tight_layout()
    plt.show()

# %% [markdown]
# #### Analisis data kategorik

# %%
for col in kategorical_features:
    if col in df.columns:  # Pastikan kolom masih ada setelah penghapusan
        print(f"\n{'='*30} Analisis Univariat Kolom Kategorikal: {col} {'='*30}")
        print(f"Jumlah Nilai Unik: {df[col].nunique()}")
        print("\nValue Counts:")
        print(df[col].value_counts())
        print("\nProporsi (%):")
        print(df[col].value_counts(normalize=True) * 100)

        plt.figure(figsize=(10, 6))
        sns.countplot(x=df[col], order=df[col].value_counts().index) # Mengurutkan berdasarkan frekuensi
        plt.title(f'Distribusi {col}')
        plt.xlabel(col)
        plt.ylabel('Frekuensi')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

# %% [markdown]
# * pada fitur application_type hanya memiliki satu nilai, jadi sebaiknya kolom ini dihapus
# * fitur earliest_cr_line bisa dianggap kurang informatif
# * fitur pymnt_plan hampir seluruh data berada dalam satu kategori, hal tersebut menunjukkan fitur tersebut memiliki varian yg rendah
# * fitur issue_d didapatkan setelah pinjaman disetujui, maka fitur ini dianggap tidak informatif untuk model
# * fitur sub_grade dan grade dihapus untuk mencegah masalah dimensi yang terlalu besar
# 

# %%
dropped_columns = ['application_type', 'earliest_cr_line', 'pymnt_plan', 'issue_d', 'sub_grade', 'grade']

df = df.drop(columns=dropped_columns, axis=1)
df.shape

# %% [markdown]
# ## Analisis multivariat

# %%
kategorical_features = df.select_dtypes(include=['object']).columns
numeric_features = df.select_dtypes(include=['number']).columns

# %%
# --- 1. Korelasi Antar Fitur Numerik ---
print("\n--- Korelasi Antar Fitur Numerik ---")
if 'loan_outcome' in numeric_features:
    numeric_features.remove('loan_outcome')

correlation_matrix = df[numeric_features].corr()
plt.figure(figsize=(18, 15))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap Korelasi Antar Fitur Numerik')
plt.show()
print("Interpretasi: Perhatikan nilai korelasi yang tinggi (positif atau negatif) antar fitur.")


# %%
# --- 2. Hubungan Antara Fitur Numerik dengan Target ---
print("\n--- Hubungan Antara Fitur Numerik dengan Target ('loan_outcome') ---")
for col in numeric_features:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=col, y='loan_outcome', data=df)
    plt.title(f'{col} vs Loan Outcome')
    plt.xlabel(col)
    plt.ylabel('Loan Outcome')
    plt.show()
print("Interpretasi: Cari pola atau tren hubungan antara fitur dan target.")

# %%
# --- 3. Hubungan Antara Fitur Kategorikal (Encoded) dengan Target ---
print("\n--- Hubungan Antara Fitur Kategorikal (Encoded) dengan Target ('loan_outcome') ---")

for col in kategorical_features:
    if df[col].nunique() < 50:  # Batasi untuk visualisasi yang lebih baik
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=col, y='loan_outcome', data=df)
        plt.title(f'{col} vs Loan Outcome')
        plt.xlabel(col)
        plt.ylabel('Loan Outcome')
        plt.show()
    else:
        print(f"Kolom '{col}' memiliki terlalu banyak nilai unik untuk box plot yang efektif.")
print("Interpretasi: Perhatikan perbedaan distribusi target di berbagai nilai fitur kategorikal.")


# %%
# --- 4. Kombinasi Dua Fitur Numerik dengan Warna untuk Target ---
print("\n--- Kombinasi Dua Fitur Numerik dengan Warna untuk Target ---")
if len(numeric_features) >= 2:
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=numeric_features[0], y=numeric_features[1], hue='loan_outcome', data=df)
    plt.title(f'{numeric_features[0]} vs {numeric_features[1]} (dengan warna berdasarkan Loan Outcome)')
    plt.xlabel(numeric_features[0])
    plt.ylabel(numeric_features[1])
    plt.show()
    print("Interpretasi: Cari pemisahan kelas target dalam ruang fitur 2D.")
else:
    print("Tidak cukup fitur numerik untuk membuat scatter plot kombinasi.")

# %% [markdown]
# # Data Preparation

# %%
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['loan_outcome_encoded'] = label_encoder.fit_transform(df['loan_outcome'])

# %%
df = df.drop('loan_outcome', axis=1)

# %%
# Identifikasi kolom kategorik (bertipe object)
categorical_cols = df.select_dtypes(include='object').columns

# One-Hot Encoding
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# %%
df.columns

# %%
from sklearn.model_selection import train_test_split

X = df.drop('loan_outcome_encoded', axis=1)
y = df['loan_outcome_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %% [markdown]
# # Data Modelling

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report



logistic_model = LogisticRegression(random_state=42, solver='liblinear')
logistic_model.fit(X_train, y_train)

# Prediksi pada data testing
y_pred_logistic = logistic_model.predict(X_test)
y_pred_proba_logistic = logistic_model.predict_proba(X_test)[:, 1]

# %% [markdown]
# # Evaluation

# %%
# Evaluasi model Logistic Regression
print("Evaluasi Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_logistic))
print("Precision:", precision_score(y_test, y_pred_logistic))
print("Recall:", recall_score(y_test, y_pred_logistic))
print("F1-Score:", f1_score(y_test, y_pred_logistic))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_proba_logistic))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_logistic))
print("\nClassification Report:\n", classification_report(y_test, y_pred_logistic))

print("\n" + "="*30 + " Random Forest " + "="*30)
# Inisialisasi dan latih model Random Forest
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)

# Prediksi pada data testing
y_pred_rf = random_forest_model.predict(X_test)
y_pred_proba_rf = random_forest_model.predict_proba(X_test)[:, 1]


# %%
# Evaluasi model Random Forest
print("Evaluasi Random Forest:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1-Score:", f1_score(y_test, y_pred_rf))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_proba_rf))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

# Opsional: Interpretasi Koefisien Logistic Regression
print("\nKoefisien Logistic Regression:")
coefficients = pd.DataFrame(logistic_model.coef_[0], index=X_train.columns, columns=['Coefficient'])
print(coefficients.sort_values(by='Coefficient', ascending=False))

# Opsional: Feature Importance Random Forest
print("\nFeature Importance Random Forest:")
feature_importances = pd.Series(random_forest_model.feature_importances_, index=X_train.columns)
print(feature_importances.sort_values(ascending=False).head(10)) # Tampilkan 10 fitur terpenting


