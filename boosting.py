import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# --- 0. Load Data ---
file_path = r"C:\Users\johan\OneDrive - Uppsala universitet\training_data_capped.xlsx"
df_capped = pd.read_excel(file_path)

# --- CLEANING: Fix Decimals ---
for col in df_capped.columns[:-1]:
    if df_capped[col].dtype == 'object':
        try:
            df_capped[col] = df_capped[col].astype(str).str.replace(',', '.').astype(float)
        except ValueError:
            pass 

# --- CLEANING: Remove Empty Targets ---
df_capped = df_capped.dropna(subset=[df_capped.columns[-1]])

# --- 1. Define Features (X) and Target (y) ---
X = df_capped.iloc[:, :-1]
y = df_capped.iloc[:, -1]

# --- 2. Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 3. Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. Train Boosting Model ---
model = GradientBoostingClassifier(random_state=42) 
model.fit(X_train_scaled, y_train)

# --- 5. Make Predictions ---
y_pred = model.predict(X_test_scaled)

# --- 6. Text Results ---
print("Model: Gradient Boosting")
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- 7. GRAPH: Confusion Matrix ---
print("Generating graph...")
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap=plt.cm.Greens)
plt.title("Boosting Confusion Matrix")
plt.show()