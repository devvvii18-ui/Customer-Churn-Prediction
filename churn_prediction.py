import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    "tenure": [12, 24, 5, 36, 8, 48, 15, 60],
    "monthly_charges": [50, 70, 90, 60, 85, 55, 75, 40],
    "total_charges": [600, 1680, 450, 2160, 680, 2640, 1125, 2400],
    "churn": [0, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df.drop("churn", axis=1)
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Customer Churn Prediction")
print("-" * 40)
print(f"Model Accuracy: {round(accuracy * 100, 2)}%")
