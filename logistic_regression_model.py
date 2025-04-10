
import random
import math

# Simulate dataset
random.seed(42)
n_samples = 1000
temperature = [random.gauss(40, 10) for _ in range(n_samples)]  # Mean 40°C, std 10
voltage = [random.gauss(220, 20) for _ in range(n_samples)]     # Mean 220V, std 20
current = [random.gauss(5, 1) for _ in range(n_samples)]        # Mean 5A, std 1
days_since_maintenance = [random.randint(0, 365) for _ in range(n_samples)]  # 0-365 days
failure = [1 if random.random() < 0.1 else 0 for _ in range(n_samples)]      # 10% failure rate

# Feature engineering: Add squared terms
X = []
for i in range(n_samples):
    t, v, c, d = temperature[i], voltage[i], current[i], days_since_maintenance[i]
    X.append([t, v, c, d, t*t, v*v])  # Add temp^2 and volt^2
y = failure

# Z-score standardization
def standardize(data):
    n_features = len(data[0])
    means = [sum(col) / len(data) for col in zip(*data)]
    stds = []
    for i in range(n_features):
        col = [row[i] for row in data]
        mean = means[i]
        variance = sum((x - mean) ** 2 for x in col) / len(data)
        stds.append(math.sqrt(variance) + 1e-6)  # Avoid division by zero
    standardized = []
    for row in data:
        norm_row = [(row[i] - means[i]) / stds[i] for i in range(n_features)]
        standardized.append(norm_row)
    return standardized

X_standardized = standardize(X)

# Split data (80% train, 20% test)
train_size = int(0.8 * n_samples)
X_train = X_standardized[:train_size]
y_train = y[:train_size]
X_test = X_standardized[train_size:]
y_test = y[train_size:]

# Logistic regression with L2 regularization
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def predict(weights, x):
    z = weights[0]  # Bias
    for i in range(len(x)):
        z += weights[i + 1] * x[i]
    return sigmoid(z)

def train_logistic(X, y, learning_rate=0.05, epochs=200, lambda_reg=0.01):
    n_features = len(X[0])
    weights = [0.0] * (n_features + 1)  # Include bias
    for _ in range(epochs):
        for i in range(len(X)):
            pred = predict(weights, X[i])
            error = pred - y[i]
            # Update bias (no regularization on bias)
            weights[0] -= learning_rate * error
            # Update weights with L2 regularization
            for j in range(n_features):
                weights[j + 1] -= learning_rate * (error * X[i][j] + lambda_reg * weights[j + 1])
    return weights

# Basic cross-validation (2-fold)
def cross_validate(X, y, learning_rate=0.05, epochs=200, lambda_reg=0.01):
    mid = len(X) // 2
    X1, X2 = X[:mid], X[mid:]
    y1, y2 = y[:mid], y[mid:]
    
    # Train on fold 1, test on fold 2
    weights1 = train_logistic(X1, y1, learning_rate, epochs, lambda_reg)
    acc1 = evaluate(X2, y2, weights1)
    # Train on fold 2, test on fold 1
    weights2 = train_logistic(X2, y2, learning_rate, epochs, lambda_reg)
    acc2 = evaluate(X1, y1, weights2)
    
    avg_acc = (acc1 + acc2) / 2
    return train_logistic(X, y, learning_rate, epochs, lambda_reg), avg_acc

def evaluate(X, y, weights, threshold=0.5):
    correct = 0
    for i in range(len(X)):
        pred_prob = predict(weights, X[i])
        pred = 1 if pred_prob >= threshold else 0
        if pred == y[i]:
            correct += 1
    return correct / len(X)

# Train with cross-validation
weights, cv_accuracy = cross_validate(X_train, y_train, learning_rate=0.05, epochs=200, lambda_reg=0.01)

# Evaluate on train and test sets
train_accuracy = evaluate(X_train, y_train, weights)
test_accuracy = evaluate(X_test, y_test, weights)

# Print results
print("Improved Model Results")
print(f"Cross-Validation Accuracy: {cv_accuracy:.4f}")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print("Learned Weights (bias, temp, volt, curr, days, temp^2, volt^2):")
print([round(w, 4) for w in weights])

# Example prediction
sample = X_test[0]
pred_prob = predict(weights, sample)
pred = 1 if pred_prob >= 0.5 else 0
print(f"\nExample Prediction:")
print(f"Sample features (standardized): {[round(x, 4) for x in sample]}")
print(f"Predicted probability: {pred_prob:.4f}")
print(f"Predicted class: {pred}, Actual class: {y_test[0]}")
