import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv, random

# Define the number of data points to generate
num_points = 10

# Define the range of x and y values
x_min, x_max = 0, 5
y_min, y_max = 0, 5

# Generate random x and y values
x_values = [random.uniform(x_min, x_max) for _ in range(num_points)]
y_values = [random.uniform(y_min, y_max) for _ in range(num_points)]

#Write the new data to the CSV file
with open('data.csv', 'w', newline='') as csvfile:
    # Declaring writer object
    writer = csv.writer(csvfile)
    # Header for csv file
    writer.writerow(['x', 'y'])
    # Write data rows using for each 
    for i in range(num_points):
        writer.writerow([x_values[i], y_values[i]])

# Load data from CSV file
data = pd.read_csv('data.csv')

# Split the data into training and test sets
# Samples random 80 % of data provided to create training set
# random state set to 1 to ensure random sampling can be reproduced
train_data = data.sample(frac=0.8, random_state=1)

test_data = data.drop(train_data.index)

# Extract the input (x) and output (y) values as NumPy arrays
x_train = train_data['x'].values
y_train = train_data['y'].values
x_test = test_data['x'].values
y_test = test_data['y'].values

# Define the learning rate and number of iterations
learning_rate = .01
num_iters = 1000

# Initialize the parameters (intercept and slope) to zero
intercept = 0
slope = 0

# Define the sum of squares cost function
def cost_function(x, y, intercept, slope):
    """
    Computes the sum of squares loss function for a linear regression model.

    Parameters:
    x (numpy array): input values
    y (numpy array): output values
    intercept (float): intercept parameter
    slope (float): slope parameter

    Returns:
    float: the cost (mean squared loss) of the linear regression model
    """
    # Calculates number of samples in the dataset
    n = len(x)
    # sum of squared residuals = (observed height - predicted height)
    residual = np.sum((y - (intercept + slope * x)) ** 2)
    # MSE Mean squared error(Measures quality of regression model)
    return residual / n

# def step_size():





# Initialize an empty list to store the cost function history
total_loss = []

# Implement gradient descent
for i in range(num_iters):
    # Compute predicted values for the training set using current parameter values
    y_pred = intercept + slope * x_train
    # Compute error between predicted values and actual values
    error = y_pred - y_train
    # Compute the cost of the current parameter values and store it in cost history
    cost = cost_function(x_train, y_train, intercept, slope)
    total_loss.append(cost)
    # Update the parameters using the gradient descent update rules
    intercept -= learning_rate * (1 / len(x_train)) * np.sum(error)
    slope -= learning_rate * (1 / len(x_train)) * np.sum(error * x_train)

# Used to define step #
step_num = 1
for nums in total_loss:
    print('Step # ', step_num, ':', nums)
    step_num+=1
print("Effective intercept:", intercept)
print("Effective slope:", slope)

# Plot the cost function history to check for convergence
plt.plot(total_loss)
plt.title("Loss Function History")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()


# Make predictions on the test set and compute the test error
y_pred_test = intercept + slope * x_test
test_error = cost_function(x_test, y_test, intercept, slope)
print("Test Error:", test_error)     