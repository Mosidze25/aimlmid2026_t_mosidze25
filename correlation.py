# Import NumPy for numerical computations (arrays, means, sums, square roots)
import numpy as np

# Import Pandas for creating a tabular representation of intermediate calculations
import pandas as pd

# Import Matplotlib for data visualization
import matplotlib.pyplot as plt


# -----------------------------
# Input Data
# -----------------------------
# Define x and y values as NumPy arrays (given coordinates)
x = np.array([-9.10, -6.80, -4.30, -2.10, 0.40, 2.70, 5.20, 7.80])
y = np.array([-7.80, -5.20, -3.10, -0.80, 1.40, 3.30, 5.70, 7.10])

# Number of observations
n = len(x)


# -----------------------------
# Compute Means
# -----------------------------
# Calculate the mean (average) of x and y values
x_mean = np.mean(x)
y_mean = np.mean(y)

# Display computed means
print("Mean of x (x̄):", x_mean)
print("Mean of y (ȳ):", y_mean)


# -----------------------------
# Compute Deviations from Mean
# -----------------------------
# Subtract the mean from each x and y value
x_dev = x - x_mean
y_dev = y - y_mean


# -----------------------------
# Compute Pearson Formula Components
# -----------------------------
# Product of deviations for numerator
product = x_dev * y_dev

# Squared deviations for denominator
x_dev_sq = x_dev ** 2
y_dev_sq = y_dev ** 2


# -----------------------------
# Create Manual Calculation Table
# -----------------------------
# Construct a DataFrame to show all intermediate values
table = pd.DataFrame({
    "x_i": x,
    "y_i": y,
    "x_i - x̄": x_dev,
    "y_i - ȳ": y_dev,
    "(x_i - x̄)(y_i - ȳ)": product,
    "(x_i - x̄)^2": x_dev_sq,
    "(y_i - ȳ)^2": y_dev_sq
})

# Display the calculation table
print("\nManual Calculation Table:\n")
print(table)


# -----------------------------
# Compute Pearson Correlation Coefficient
# -----------------------------
# Numerator: sum of products of deviations
numerator = np.sum(product)

# Denominator: square root of the product of squared deviations
denominator = np.sqrt(np.sum(x_dev_sq) * np.sum(y_dev_sq))

# Pearson correlation coefficient
r = numerator / denominator

# Display intermediate sums and final result
print("\nΣ(x_i - x̄)(y_i - ȳ) =", numerator)
print("Σ(x_i - x̄)^2 =", np.sum(x_dev_sq))
print("Σ(y_i - ȳ)^2 =", np.sum(y_dev_sq))
print("\nPearson Correlation Coefficient (r) =", r)


# -----------------------------
# Visualization
# -----------------------------
# Create a scatter plot of the data points
plt.figure()
plt.scatter(x, y)

# Compute line of best fit using least squares
m, b = np.polyfit(x, y, 1)

# Plot the regression line
plt.plot(x, m * x + b)

# Label axes and add title
plt.xlabel("x")
plt.ylabel("y")
plt.title("Scatter Plot with Line of Best Fit")

# Display the plot
plt.show()
