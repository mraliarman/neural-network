import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

class KNNModel(nn.Module):
    def __init__(self, k=1, distance_metric='euclidean'):
        super(KNNModel, self).__init__()
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = self.calculate_distances(x)
            _, indices = torch.topk(distances, self.k, largest=False)
            k_nearest_labels = self.y_train[indices]
            unique_labels, counts = torch.unique(k_nearest_labels, return_counts=True)
            predicted_label = unique_labels[torch.argmax(counts)]
            predictions.append(predicted_label.item())
        return torch.tensor(predictions)

    def calculate_distances(self, x):
        if self.distance_metric == 'euclidean':
            distances = torch.norm(self.X_train - x, dim=1, p=2)
        elif self.distance_metric == 'manhattan':
            distances = torch.norm(self.X_train - x, dim=1, p=1)
        return distances

class MLPModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, hidden_neurons, activation_function):
        super(MLPModel, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_neurons))
        layers.append(activation_function())
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(activation_function())
            
        layers.append(nn.Linear(hidden_neurons, output_size))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
    
#step 1: opening file
def read_dataset(file_path):
    return pd.read_csv(file_path)

# Step 2: Encode categorical data using LabelEncoder
def encode_categorical_data(data):
    label_encoder = LabelEncoder()
    for column in data.select_dtypes(include=['object']).columns:
        data[column] = label_encoder.fit_transform(data[column])
    return data

# Step 3: Handle missing values by replacing with the mean
def handle_missing_values(data):
    data_filled = data.fillna(data.mean())
    return data_filled

# Step 4: Plot the Success Rate
def plot_success_rate(success_rate_train, success_rate_val, hidden_layers, hidden_neurons, activation_function):
    plt.plot(success_rate_train, label=f'Training Success Rate')
    plt.plot(success_rate_val, label=f'Validation Success Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Success Rate')
    plt.title(f'Hidden Layers: {hidden_layers} hidden neurons: {hidden_neurons}, activation function: {activation_function}')
    plt.legend()
    plt.show()
    
# Step 5: Train and Evaluate the KNN Model
def train_and_evaluate_knn_model(k_values, distance_metrics):
    for distance_metric in distance_metrics:
        train_accuracy = []
        test_accuracy = []

        for k in k_values:
            knn_model = KNNModel(k=k, distance_metric=distance_metric)
            knn_model.fit(torch.Tensor(X_train.values), torch.LongTensor(y_train.values))

            y_pred_train = knn_model.predict(torch.Tensor(X_train.values))
            y_pred_test = knn_model.predict(torch.Tensor(X_test.values))

            train_accuracy.append(accuracy_score(y_train, y_pred_train))
            test_accuracy.append(accuracy_score(y_test, y_pred_test))

            print(f"\nKNN with k={k} and distance metric={distance_metric}:")
            print("Training Classification Report:\n", classification_report(y_train, y_pred_train))
            print("Testing Classification Report:\n", classification_report(y_test, y_pred_test))

        # Plot accuracy vs. k for both training and testing sets
        plt.plot(k_values, train_accuracy, label=f'Training Accuracy (Distance Metric: {distance_metric})')
        plt.plot(k_values, test_accuracy, label=f'Testing Accuracy (Distance Metric: {distance_metric})')

        plt.xlabel('k')
        plt.ylabel('Accuracy')
        plt.title(f'KNN Model Accuracy for different K Values and {distance_metric}')
        plt.legend()
        plt.show()

# Step 6: Train the MLP Model
def train_mlp_model(model, criterion, optimizer, train_loader, val_loader, epochs=20):
    success_rate_train = []
    success_rate_val = []

    for epoch in range(epochs):
        model.train()
        total_train = 0
        correct_train = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        success_rate_train.append(correct_train / total_train)

        model.eval()
        total_val = 0
        correct_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

            success_rate_val.append(correct_val / total_val)

        print(f'Epoch {epoch + 1}/{epochs}, Training Success Rate: {success_rate_train[-1]}, Validation Success Rate: {success_rate_val[-1]}')

    return success_rate_train, success_rate_val

# Step 7: Train and Evaluate the MLP Model
def train_and_evaluate_mlp_model(hidden_layers_list, hidden_neurons_list, activation_functions):
    best_model = None
    best_val_accuracy = 0
    best_hidden_layers = 0
    best_hidden_neurons = 0
    best_activation_function = None

    for hidden_layers in hidden_layers_list:
        for hidden_neurons in hidden_neurons_list:
            for activation_function in activation_functions:
                # Build and train the MLP model
                mlp_model = MLPModel(input_size=len(X_train.columns), output_size=len(y_train.unique()), hidden_layers=hidden_layers, hidden_neurons=hidden_neurons, activation_function=activation_function)
                criterion = nn.CrossEntropyLoss()
                # optimizer = optim.Adam(mlp_model.parameters(), lr=0.0000001)
                optimizer = optim.SGD(mlp_model.parameters(), lr=0.01, momentum=0.9)


                success_rate_train, success_rate_val = train_mlp_model(mlp_model, criterion, optimizer, train_loader, val_loader)

                # plot_success_rate(success_rate_train, success_rate_val, hidden_layers, hidden_neurons, activation_function.__name__)

                # Evaluate the MLP Model on validation set
                mlp_model.eval()
                with torch.no_grad():
                    y_pred_val_mlp = torch.argmax(mlp_model(X_val_tensor), axis=1)
                    val_accuracy = torch.sum(y_pred_val_mlp == y_val_tensor) / len(y_val_tensor)

                    print(f"\nMLP Model Classification Report - Validation Set (Hidden Layers: {hidden_layers}, Hidden Neurons: {hidden_neurons}, Activation Function: {activation_function.__name__}):\n", classification_report(y_val_tensor, y_pred_val_mlp))

                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_model = mlp_model
                    best_hidden_layers = hidden_layers
                    best_hidden_neurons = hidden_neurons
                    best_activation_function = activation_function

    print(f"\nBest MLP Model was trained with {best_hidden_layers} hidden layers, {best_hidden_neurons} hidden neurons, and {best_activation_function.__name__} activation function.")

    return best_model

    
file_path = 'Telecust1.csv'
data = read_dataset(file_path)
data_encoded = encode_categorical_data(data)
data_filled = handle_missing_values(data_encoded)

# split atribut and labels
X = data_filled.iloc[:, 1:-1]
y = data_filled['custcat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17, shuffle=False)

k_values = [1, 3, 5, 7]
distance_metrics = ['euclidean', 'manhattan']
# train_and_evaluate_knn_model(k_values, distance_metrics)

# Prepare data for MLP
X_train_tensor = torch.Tensor(X_train.values)
y_train_tensor = torch.LongTensor(y_train.values)
X_val_tensor = torch.Tensor(X_test.values)
y_val_tensor = torch.LongTensor(y_test.values)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Hidden layers, neurons, and activation functions to try
hidden_layers_to_try = [2, 3, 4, 5, 6]
hidden_neurons_to_try = [32, 64, 128, 256, 512]
activation_functions_to_try = [nn.LeakyReLU, nn.Tanh, nn.ReLU, nn.Sigmoid]
# hidden_layers_to_try = [2]
# hidden_neurons_to_try = [64]
# activation_functions_to_try = [nn.Tanh]
best_mlp_model = train_and_evaluate_mlp_model(hidden_layers_to_try, hidden_neurons_to_try, activation_functions_to_try)

# Evaluate the best MLP model on the testing set
best_mlp_model.eval()
with torch.no_grad():
    y_pred_test_mlp = torch.argmax(best_mlp_model(X_val_tensor), axis=1)
    print("\nBest MLP Model Classification Report - Testing Set:\n", classification_report(y_val_tensor, y_pred_test_mlp))

# Best MLP Model was trained with 4 hidden layers, 256 hidden neurons, and ReLU activation function.
# Best MLP Model Classification Report - Testing Set:
#                precision    recall  f1-score   support

#            0       0.35      0.60      0.44        45
#            1       0.37      0.29      0.33        34
#            2       0.54      0.70      0.61        69
#            3       0.17      0.02      0.03        52

#     accuracy                           0.43       200
#    macro avg       0.36      0.40      0.35       200
# weighted avg       0.37      0.43      0.37       200



# Best MLP Model was trained with 5 hidden layers, 512 hidden neurons, and Tanh activation function.
# Best MLP Model Classification Report - Testing Set:
#                precision    recall  f1-score   support

#            0       0.47      0.58      0.52        53
#            1       0.35      0.39      0.37        41
#            2       0.51      0.50      0.50        64
#            3       0.32      0.19      0.24        42

#     accuracy                           0.43       200
#    macro avg       0.41      0.42      0.41       200
# weighted avg       0.43      0.43      0.42       200



# Best MLP Model was trained with 2 hidden layers, 256 hidden neurons, and LeakyReLU activation function.
# Best MLP Model Classification Report - Testing Set:
#                precision    recall  f1-score   support

#            0       0.40      0.68      0.50        53
#            1       0.80      0.10      0.17        41
#            2       0.48      0.55      0.51        64
#            3       0.31      0.24      0.27        42

#     accuracy                           0.42       200
#    macro avg       0.50      0.39      0.36       200
# weighted avg       0.49      0.42      0.39       200



# Best MLP Model was trained with 2 hidden layers, 512 hidden neurons, and LeakyReLU activation function.
# Best MLP Model Classification Report - Testing Set:
#                precision    recall  f1-score   support

#            0       0.36      0.67      0.47        45
#            1       0.33      0.03      0.05        34
#            2       0.46      0.71      0.56        69
#            3       0.67      0.08      0.14        52

#     accuracy                           0.42       200
#    macro avg       0.45      0.37      0.30       200
# weighted avg       0.47      0.42      0.34       200



# Best MLP Model was trained with 2 hidden layers, 128 hidden neurons, and Sigmoid activation function.
# Best MLP Model Classification Report - Testing Set:
#                precision    recall  f1-score   support

#            0       0.36      0.60      0.45        45
#            1       0.33      0.47      0.39        34
#            2       0.72      0.42      0.53        69
#            3       0.34      0.25      0.29        52

#     accuracy                           0.42       200
#    macro avg       0.44      0.44      0.42       200
# weighted avg       0.48      0.42      0.43       200