
import random

# Simulate dataset (temperature, voltage, current, days_since_maintenance, failure)
random.seed(42)  # For reproducibility
n_samples = 1000

# Generate synthetic data
temperature = [random.gauss(40, 10) for _ in range(n_samples)]  # Mean 40°C, std 10
voltage = [random.gauss(220, 20) for _ in range(n_samples)]     # Mean 220V, std 20
current = [random.gauss(5, 1) for _ in range(n_samples)]        # Mean 5A, std 1
days_since_maintenance = [random.randint(0, 365) for _ in range(n_samples)]  # 0-365 days
failure = [1 if random.random() < 0.1 else 0 for _ in range(n_samples)]      # 10% failure rate

# Organize features into a dictionary
features = {
    'temperature': temperature,
    'voltage': voltage,
    'current': current,
    'days_since_maintenance': days_since_maintenance
}

# Calculate importance: absolute difference in averages between failure and no-failure
print("Feature Importance (Absolute Difference in Averages):")
for feature_name, feature_values in features.items():
    # Split values by failure status
    fail_vals = [feature_values[i] for i in range(n_samples) if failure[i] == 1]
    no_fail_vals = [feature_values[i] for i in range(n_samples) if failure[i] == 0]
    
    # Compute averages
    avg_fail = sum(fail_vals) / len(fail_vals) if fail_vals else 0
    avg_no_fail = sum(no_fail_vals) / len(no_fail_vals) if no_fail_vals else 0
    
    # Importance is the absolute difference
    importance = abs(avg_fail - avg_no_fail)
    print(f"{feature_name}: {importance:.4f}")

# Rank features by importance
importance_dict = {}
for feature_name, feature_values in features.items():
    fail_vals = [feature_values[i] for i in range(n_samples) if failure[i] == 1]
    no_fail_vals = [feature_values[i] for i in range(n_samples) if failure[i] == 0]
    avg_fail = sum(fail_vals) / len(fail_vals) if fail_vals else 0
    avg_no_fail = sum(no_fail_vals) / len(no_fail_vals) if no_fail_vals else 0
    importance_dict[feature_name] = abs(avg_fail - avg_no_fail)

# Sort and display ranked features
print("\nRanked Features by Importance:")
sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
for feature, importance in sorted_features:
    print(f"{feature}: {importance:.4f}")
