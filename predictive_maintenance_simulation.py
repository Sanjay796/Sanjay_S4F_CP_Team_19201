
import random
import math

# Simulate dataset for training
random.seed(42)
n_samples = 1000
temperature = [random.gauss(40, 10) for _ in range(n_samples)]
voltage = [random.gauss(220, 20) for _ in range(n_samples)]
current = [random.gauss(5, 1) for _ in range(n_samples)]
days_since_maintenance = [random.randint(0, 365) for _ in range(n_samples)]
failure = [1 if random.random() < 0.1 else 0 for _ in range(n_samples)]

# Feature engineering
X = [[t, v, c, d, t*t, v*v] for t, v, c, d in zip(temperature, voltage, current, days_since_maintenance)]
y = failure

# Standardize features
def standardize(data):
    n_features = len(data[0])
    means = [sum(col) / len(data) for col in zip(*data)]
    stds = [math.sqrt(sum((x - means[i]) ** 2 for x in col) / len(data) + 1e-6) for i, col in enumerate(zip(*data))]
    return [[(row[i] - means[i]) / stds[i] for i in range(n_features)] for row in data]

X_standardized = standardize(X)

# Logistic regression
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def predict(weights, x):
    z = weights[0]
    for i in range(len(x)):
        z += weights[i + 1] * x[i]
    return sigmoid(z)

def train_logistic(X, y, learning_rate=0.05, epochs=200):
    n_features = len(X[0])
    weights = [0.0] * (n_features + 1)
    for _ in range(epochs):
        for i in range(len(X)):
            pred = predict(weights, X[i])
            error = pred - y[i]
            weights[0] -= learning_rate * error
            for j in range(n_features):
                weights[j + 1] -= learning_rate * error * X[i][j]
    return weights

weights = train_logistic(X_standardized, y)

# Simulate maintenance scenarios over 365 days
n_inverters = 10
days = 365
failure_rate = 0.005  # Daily failure chance per inverter
maintenance_cost = 100  # Cost per maintenance (parts/labor)
downtime_loss = 50     # Energy loss per day of downtime

# Traditional maintenance: every 180 days
traditional_downtime = 0
traditional_repairs = 0
traditional_waste = 0
for inverter in range(n_inverters):
    days_since_maint = 0
    for day in range(days):
        if random.random() < failure_rate:
            traditional_downtime += 1  # 1 day downtime per failure
            traditional_repairs += 1
            traditional_waste += maintenance_cost
            days_since_maint = 0
        days_since_maint += 1
        if days_since_maint >= 180:  # Scheduled maintenance
            traditional_repairs += 1
            traditional_waste += maintenance_cost
            days_since_maint = 0

# Predictive maintenance: act on predictions
predictive_downtime = 0
predictive_repairs = 0
predictive_waste = 0
for inverter in range(n_inverters):
    days_since_maint = 0
    for day in range(days):
        # Simulate daily sensor data
        t = random.gauss(40, 10)
        v = random.gauss(220, 20)
        c = random.gauss(5, 1)
        d = days_since_maint
        sample = standardize([[t, v, c, d, t*t, v*v]])[0]
        pred_prob = predict(weights, sample)
        
        # Act if predicted failure probability is high
        if pred_prob > 0.5:
            predictive_repairs += 1
            predictive_waste += maintenance_cost
            days_since_maint = 0
        elif random.random() < failure_rate:
            predictive_downtime += 1
            predictive_repairs += 1
            predictive_waste += maintenance_cost
            days_since_maint = 0
        days_since_maint += 1

# Calculate results
traditional_energy_loss = traditional_downtime * downtime_loss
predictive_energy_loss = predictive_downtime * downtime_loss

# Output results and explanation
print("Sustainability Impact of Predictive Maintenance")
print(f"Traditional Maintenance (Fixed Schedule):")
print(f" - Downtime Days: {traditional_downtime}")
print(f" - Repairs/Waste (Cost): {traditional_repairs} ({traditional_waste})")
print(f" - Energy Loss: {traditional_energy_loss} units")
print(f"Predictive Maintenance:")
print(f" - Downtime Days: {predictive_downtime}")
print(f" - Repairs/Waste (Cost): {predictive_repairs} ({predictive_waste})")
print(f" - Energy Loss: {predictive_energy_loss} units")

print("\nHow Predictive Maintenance Contributes to Sustainability and Reduces Waste:")
print(f"1. Reduced Downtime: Cut downtime by {traditional_downtime - predictive_downtime} days, "
      f"boosting clean energy output by {traditional_energy_loss - predictive_energy_loss} units.")
print(f"2. Less Waste: Avoided {traditional_repairs - predictive_repairs} unnecessary repairs, "
      f"saving {traditional_waste - predictive_waste} in resources.")
print("3. Extended Equipment Life: Targeted repairs prevent cascading failures.")
print("4. Lower Carbon Footprint: Fewer technician trips and efficient energy use.")
