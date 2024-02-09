# مرحلە ۱: خواندن داده و پیش‌پردازش
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# خواندن داده
data = pd.read_csv('Telecust1.csv')

# پیش‌پردازش داده
X = data.drop('custcat', axis=1)
y = data['custcat']

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
print("گزارش دقت برچسب‌های آموزشی:\n", classification_report(y_train, y_pred_train))
print("گزارش دقت برچسب‌های آزمایشی:\n", classification_report(y_test, y_pred_test))

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

# مرحلە ٣: شبکه‌ی عصبی چندلایه (MLP)
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# تعریف تابع برای آنالیز مدل MLP با تعداد لایه‌ها و نورون‌های مختلف
def analyze_mlp_model(hidden_layer_sizes_values, activation_function):
    train_scores = []
    test_scores = []
    
    for hidden_layer_sizes in hidden_layer_sizes_values:
        mlp_model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=20, random_state=17)
        mlp_model.fit(X_train, y_train)
        train_score = mlp_model.score(X_train, y_train)
        test_score = mlp_model.score(X_test, y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)
    
    # رسم نمودار درصد موفقیت پیش‌بینی مدل
    plt.figure(figsize=(8, 4))
    plt.plot(hidden_layer_sizes_values, train_scores, label='Training Accuracy', marker='o')
    plt.plot(hidden_layer_sizes_values, test_scores, label='Testing Accuracy', marker='o')
    plt.title(f'MLP with {activation_function} Activation Function')
    plt.xlabel('Hidden Layer Sizes')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# اجرای تابع برای تعداد لایه‌ها مختلف با تابع فعال‌سازی ReLU
hidden_layer_sizes_values = [(28,), (28, 28)]
analyze_mlp_model(hidden_layer_sizes_values, 'ReLU')

# مرحلە ٤: انتخاب بهترین مدل MLP و آموزش با تعداد نورون‌های مختلف
best_mlp_model = MLPClassifier(hidden_layer_sizes=(28,), max_iter=20, random_state=17)

neuron_values = [32, 63, 128, 256, 512]
analyze_mlp_model([(28, n) for n in neuron_values], 'ReLU')

# مرحلە ٥: انتخاب بهترین مدل MLP و آموزش با توابع فعال‌سازی مختلف
activation_functions = ['leaky_relu', 'tanh', 'relu', 'sigmoid']

for activation_function in activation_functions:
    mlp_model = MLPClassifier(hidden_layer_sizes=(28,), activation=activation_function, max_iter=20, random_state=17)
    mlp_model.fit(X_train, y_train)
    
    print(f"\nMLP with {activation_function} Activation Function:")
    print("Training Classification Report:\n", classification_report(y_train, mlp_model.predict(X_train)))
    print("Testing Classification Report:\n", classification_report(y_test, mlp_model.predict(X_test)))

# مرحلە ٦: آموزش مدل با بهینه‌گرهای مختلف
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer

# تعریف توابع بهینه‌گر و آموزش مدل
def train_model_with_optimizer(model, optimizer, learning_rate_values):
    train_scores = []
    test_scores = []
    
    for learning_rate in learning_rate_values:
        model.set_params(optimizer__learning_rate_init=learning_rate)
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)
    
    # رسم نمودار درصد موفقیت پیش‌بینی مدل
    plt.figure(figsize=(8, 4))
    plt.plot(learning_rate_values, train_scores, label='Training Accuracy', marker='o')
    plt.plot(learning_rate_values, test_scores, label='Testing Accuracy', marker='o')
    plt.title(f'Model with {optimizer} Optimizer')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# اجرای توابع برای بهینه‌گر Adam
adam_optimizer = MLPClassifier(hidden_layer_sizes=(28,), max_iter=20, random_state=17, solver='adam', random_seed=17)
learning_rate_values_adam = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
train_model_with_optimizer(adam_optimizer, 'Adam', learning_rate_values_adam)

# اجرای توابع برای بهینه‌گر SGD
sgd_optimizer = MLPClassifier(hidden_layer_sizes=(28,), max_iter=20, random_state=17, solver='sgd', random_seed=17)
learning_rate_values_sgd = [1e-1, 1e-2, 1e-3, 1e-4]
train_model_with_optimizer(sgd_optimizer, 'SGD', learning_rate_values_sgd)

# مرحلە ٧: مقایسه بهینه‌گرهای Adam و SGD
# انتخاب بهترین مدل از مرحله ٦
best_optimizer_model = adam_optimizer  # یا می‌توان انتخاب کرد sgd_optimizer
best_learning_rate = 1e-4  # بر اساس نتایج به دست آمده از مرحله ٦

# آموزش مدل با بهترین تنظیمات
best_optimizer_model.set_params(optimizer__learning_rate_init=best_learning_rate)
best_optimizer_model.fit(X_train, y_train)

print("\nBest Model with Adam Optimizer:")
print("Training Classification Report:\n", classification_report(y_train, best_optimizer_model.predict(X_train)))
print("Testing Classification Report:\n", classification_report(y_test, best_optimizer_model.predict(X_test)))

# مقایسه دقت بهینه‌گرهای Adam و SGD
adam_scores = cross_val_score(adam_optimizer, X_train, y_train, cv=StratifiedKFold(n_splits=5), scoring=make_scorer(lambda model, X, y: model.score(X, y)))
sgd_scores = cross_val_score(sgd_optimizer, X_train, y_train, cv=StratifiedKFold(n_splits=5), scoring=make_scorer(lambda model, X, y: model.score(X, y)))

print("\nAdam Optimizer Cross-Validation Scores:", adam_scores)
print("SGD Optimizer Cross-Validation Scores:", sgd_scores)
