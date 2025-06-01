import numpy as np  # For numerical operations
import pandas as pd  # For data processing

# Load the dataset
path = "Salary_dataset.csv"
DataFrame = pd.read_csv(path)

# Feature and target selection
x = DataFrame[["YearsExperience"]]
y = DataFrame["Salary"]

# Train-test split
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, shuffle=True)

# Import plotting library
import matplotlib.pyplot as plt

# 1) Scatter plot of full data WITHOUT regression line
plt.scatter(x, y)
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Years of Experience")
plt.show()

# Train the linear regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train_x, train_y)

# 2) Scatter plot of full data WITH regression line
predicted_values = model.predict(x)
plt.scatter(x, y)
plt.plot(x, predicted_values, color="red", label="Regression Line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Regression Line Fit")
plt.legend()
plt.show()

# Predict on test set
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
y_predicted = model.predict(test_x)

# Evaluation metrics
r2 = r2_score(test_y, y_predicted)
mae = mean_absolute_error(test_y, y_predicted)
mse = mean_squared_error(test_y, y_predicted)

print("Model Evaluation:")
print("R² Score:", r2)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)

# Compare actual vs predicted values
data = {
    'Actual Salary': test_y,
    'Predicted Salary': y_predicted,
}
df = pd.DataFrame(data)
print("\nActual vs Predicted Salaries:")
print(df)

# Save predictions to CSV
df.to_csv("predictions.csv", index=False)

# Predict for a custom input (wrapped in DataFrame to avoid warning)
custom_exp = pd.DataFrame([[5]], columns=["YearsExperience"])
custom_pred = model.predict(custom_exp)
print(f"\nPredicted salary for 5 years of experience: {custom_pred[0]:.2f}")
