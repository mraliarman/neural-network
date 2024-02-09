import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def read_dataset(file_path):
    return pd.read_csv(file_path)

# Step 2: Encode categorical data using LabelEncoder
def encode_categorical_data(data):
    label_encoder = LabelEncoder()
    for column in data.select_dtypes(include=['object']).columns:
        data[column] = label_encoder.fit_transform(data[column])
    return data

# Step 3: Handle missing values by dropping or replacing with the mean
def handle_missing_values(data):
    # First method: Delete rows with NaN values
    data_dropped = data.dropna()
    
    # Second method: Replace NaN values with the mean of the column
    data_filled = data.fillna(data.mean())
    
    # Report the difference in the number of data
    difference = len(data) - len(data_dropped)
    print(f"Difference in the number of data after handling missing values: {difference}")
    
    return data_filled


file_path = 'Telecust1.csv'
# file_path = 'Telecust1-Null.csv'
data = read_dataset(file_path)
data_encoded = encode_categorical_data(data)
data_filled = handle_missing_values(data_encoded)
# خواندن داده

# جدا کردن ویژگی‌ها و برچسب‌ها
X = data_filled.iloc[:, 1:-1]
y = data_filled['custcat']


# تعیین ستون‌های عددی و دسته‌ای
numeric_features = X.select_dtypes(include=['number']).columns
categorical_features = X.select_dtypes(exclude=['number']).columns

# ایجاد یک پایپلاین برای پیش‌پردازش
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# افزودن پیش‌پردازش به مدل
knn_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier())
])

# تقسیم داده به داده‌های آموزشی و آزمایشی
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17, shuffle=True)

# آموزش مدل و ارزیابی
knn_model.fit(X_train, y_train)
y_pred_train = knn_model.predict(X_train)
y_pred_test = knn_model.predict(X_test)

# گزارش نتایج
print("Training Classification Report:\n", classification_report(y_train, y_pred_train))
print("Testing Classification Report:\n", classification_report(y_test, y_pred_test))


# مرحلە ٢: الگوریتم K-نزدیکترین همسایه (KNN)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# تعریف تابع برای آنالیز مدل KNN با مقادیر k مختلف
def analyze_knn_model(k_values):
    for k in k_values:
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(X_train, y_train)
        y_pred_train = knn_model.predict(X_train)
        y_pred_test = knn_model.predict(X_test)
        
        print(f"\nKNN with k={k}:")
        print("Training Classification Report:\n", classification_report(y_train, y_pred_train))
        print("Testing Classification Report:\n", classification_report(y_test, y_pred_test))

# اجرای تابع برای مقادیر k مختلف
k_values = [1, 3, 5, 7]
analyze_knn_model(k_values)

# تبدیل برچسب‌ها به اعداد
y_train_nn = pd.factorize(y_train)[0]
y_test_nn = pd.factorize(y_test)[0]

# تبدیل داده‌ها به tensor در pytorch
X_train_tensor = torch.Tensor(X_train.values)
y_train_tensor = torch.LongTensor(y_train_nn)
X_test_tensor = torch.Tensor(X_test.values)
y_test_tensor = torch.LongTensor(y_test_nn)




# تعریف مدل شبکه عصبی چندلایه
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 28)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(28, len(data['custcat'].unique()))  # تعداد خروجی‌ها برابر با تعداد دسته‌ها

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# تعریف مدل، تابع هزینه و بهینه‌ساز
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# تبدیل داده‌ها به شیوه‌ی DataLoader در pytorch
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# آموزش مدل
epochs = 50
for epoch in range(epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# ارزیابی مدل
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    
    # نمایش نتایج
    print("MLP Classifier")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_nn, predicted))
    print("Classification Report:")
    print(classification_report(y_test_nn, predicted))
