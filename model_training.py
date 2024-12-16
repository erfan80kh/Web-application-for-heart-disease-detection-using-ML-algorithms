import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

def load_and_preprocess_data(filepath):
    """بارگذاری و پیش‌پردازش دیتاست شامل کدگذاری و تقسیم داده‌ها."""
    data = pd.read_csv(filepath)
    data = pd.get_dummies(data)
    X, y = data.drop(columns='target'), data['target']
    return X, y

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """ایجاد، آموزش، و ارزیابی مدل Random Forest."""
    model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    
    # بررسی مدل
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return model

def save_model_and_scaler(model, scaler, features):
    """ذخیره مدل، مقیاس‌کننده، و ویژگی‌ها برای استفاده در برنامه Flask."""
    joblib.dump(model, 'heart_disease_model_random_forest.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(features, 'features.pkl')
    print("Model, scaler, and features saved successfully.")

# بارگذاری داده‌ها و پیش‌پردازش
X, y = load_and_preprocess_data('heart.csv')
features = X.columns

# تقسیم داده‌ها و نرمال‌سازی
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# آموزش و ذخیره مدل
model = train_and_evaluate_model(X_train, X_test, y_train, y_test)
save_model_and_scaler(model, scaler, features)
