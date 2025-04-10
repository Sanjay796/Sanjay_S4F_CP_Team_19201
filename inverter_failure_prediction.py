
import random
import math

# Simulate dataset (for training the model)
random.seed(42)
n_samples = 1000
temperature = [random.gauss(40, 10) for _ in range(n_samples)]
voltage = [random.gauss(220, 20) for _ in range(n_samples)]
current = [random.gauss(5, 1) for _ in range(n_samples)]
days_since_maintenance = [random.randint(0, 365) for _ in range(n_samples)]
failure = [1 if random.random() < 0.1 else 0 for _ in range(n_samples)]

# Feature engineering: Add squared terms
X = [[t, v, c, d, t*t, v*v] for t, v, c, d in zip(temperature, voltage, current, days_since_maintenance)]
y = failure

# Z-score standardization
def standardize(data):
    n_features = len(data[0])
    means = [sum(col) / len(data) for col in zip(*data)]
    stds = [math.sqrt(sum((x - means[i]) ** 2 for x in col) / len(data) + 1e-6) for i, col in enumerate(zip(*data))]
    return [[(row[i] - means[i]) / stds[i] for i in range(n_features)] for row in data]

X_standardized = standardize(X)

# Logistic regression functions
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def predict(weights, x):
    z = weights[0]  # Bias
    for i in range(len(x)):
        z += weights[i + 1] * x[i]
    return sigmoid(z)

def train_logistic(X, y, learning_rate=0.05, epochs=200, lambda_reg=0.01):
    n_features = len(X[0])
    weights = [0.0] * (n_features + 1)
    for _ in range(epochs):
        for i in range(len(X)):
            pred = predict(weights, X[i])
            error = pred - y[i]
            weights[0] -= learning_rate * error
            for j in range(n_features):
                weights[j + 1] -= learning_rate * (error * X[i][j] + lambda_reg * weights[j + 1])
    return weights

# Train the model (using all data for simplicity)
weights = train_logistic(X_standardized, y)

# Define action rules based on prediction
def recommend_actions(pred_prob, sample_features):
    print(f"Predicted Failure Probability: {pred_prob:.4f}")
    print(f"Sample Features (standardized): {[round(x, 4) for x in sample_features]}")
    
    if pred_prob > 0.7:
        action = "Urgent Action Required"
        details = (
            "- Immediate inspection of the inverter is recommended.\n"
            "- Preemptive replacement may be necessary if critical components (e.g., cooling system) are failing.\n"
            "- Shut down the inverter during off-peak hours to avoid unplanned downtime."
        )
    elif pred_prob > 0.5:
        action = "Schedule Maintenance Soon"
        details = (
            "- Plan a technician visit within the next 1-3 days.\n"
            "- Focus on checking temperature regulation and electrical connections.\n"
            "- Increase sensor monitoring frequency (e.g., every 5 minutes) to track progression."
        )
    else:
        action = "Monitor, No Immediate Action"
        details = (
            "- Continue regular monitoring with current sensor schedule.\n"
            "- Log this prediction for future model retraining.\n"
            "- No maintenance needed unless new anomalies appear."
        )
    
    print(f"Recommended Action: {action}")
    print("Details:")
    print(details)

# Test with a sample (e.g., first test sample after split)
train_size = int(0.8 * n_samples)
X_test = X_standardized[train_size:]
sample = X_test[0]  # First test sample
pred_prob = predict(weights, sample)

# Recommend actions
recommend_actions(pred_prob, sample)
