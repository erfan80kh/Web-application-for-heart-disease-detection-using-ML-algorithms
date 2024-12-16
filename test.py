import pandas as pd

# بارگذاری داده‌ها
data = pd.read_csv('heart.csv')

# نمایش نام ستون‌ها
print("Before get_dummies:", data.columns.tolist())

# تبدیل رشته‌ها به عدد با استفاده از pd.get_dummies()
data = pd.get_dummies(data, drop_first=True)

# نمایش نام ستون‌ها بعد از تبدیل
print("After get_dummies:", data.columns.tolist())