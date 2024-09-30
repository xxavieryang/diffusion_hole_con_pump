import pandas as pd
import matplotlib.pyplot as plt

# Read data from two CSV files
data1 = pd.read_csv('data1.csv')  # Replace with your first CSV filename
data2 = pd.read_csv('data2.csv')  # Replace with your second CSV filename

# Assuming each CSV has 'x' and 'y' columns
# Adjust the column names based on your CSV structure
x1 = data1['x']
y1 = data1['y']

x2 = data2['x']
y2 = data2['y']

# Create a plot
plt.figure(figsize=(10, 5))

# Plot data from the first CSV
plt.plot(x1, y1, label='Data from CSV 1', color='blue', marker='o')

# Plot data from the second CSV
plt.plot(x2, y2, label='Data from CSV 2', color='orange', marker='x')

# Add labels and title
plt.xlabel('X-axis Label')  # Replace with your X-axis label
plt.ylabel('Y-axis Label')  # Replace with your Y-axis label
plt.title('Comparison of Two CSV Data')

# Add a legend
plt.legend()

# Show grid
plt.grid()

# Show the plot
plt.show()