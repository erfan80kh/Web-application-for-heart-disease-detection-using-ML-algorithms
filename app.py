import pandas as pd
import joblib
from flask import Flask, render_template, request

# بارگذاری مدل‌ها و ویژگی‌ها
model = joblib.load('heart_disease_model_random_forest.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')  # لیست ویژگی‌های مدل

app = Flask(__name__)

def preprocess_input(form_data):
    """داده‌های ورودی را از فرم دریافت کرده و آن‌ها را به فرمت موردنیاز مدل تبدیل می‌کند."""
    input_data = {
        'age': float(form_data.get('age', 0)),
        'resting_blood_pressure': float(form_data.get('resting_blood_pressure', 0)),
        'cholesterol': float(form_data.get('cholesterol', 0)),
        'Max_heart_rate': float(form_data.get('Max_heart_rate', 0)),
        'oldpeak': float(form_data.get('oldpeak', 0))
    }

    # ایجاد ویژگی‌های دسته‌بندی شده با مقداردهی اولیه صفر
    categories = {
        'sex': ['Female', 'Male'],
        'cp': ['Asymptomatic', 'Atypical angina', 'Non-anginal pain', 'Typical angina'],
        'fbs': ['Greater than 120 mg/ml', 'Lower than 120 mg/ml'],
        'restecg': ['Left ventricular hypertrophy', 'Normal', 'ST-T wave abnormality'],
        'exang': ['No', 'Yes'],
        'slope': ['Downsloping', 'Flat', 'Upsloping'],
        'ca': ['Zero', 'One', 'Two', 'Three', 'Four'],
        'thal': ['Normal', 'Fixed Defect', 'Reversable Defect']
    }

    # مقداردهی ویژگی‌های دسته‌بندی شده بر اساس فرم
    for cat, values in categories.items():
        for value in values:
            input_data[f"{cat}_{value}"] = 1.0 if form_data.get(cat) == value else 0.0

    # تبدیل به DataFrame و مرتب‌سازی بر اساس ویژگی‌های مدل
    input_df = pd.DataFrame([input_data]).reindex(columns=features, fill_value=0)
    return scaler.transform(input_df)  

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_scaled = preprocess_input(request.form)
    prediction = model.predict(input_scaled)[0]
    result = "بیمار" if prediction == 1 else "سالم"
    return render_template('result.html', result=result)



if __name__ == '__main__':
    app.run(debug=True)
