# linear-regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\vinay\Downloads\data.csv.csv")  

data["X"] = data["X"].astype(float)
data["Y"] = data["Y"].astype(float)

def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].X
        y = points.iloc[i].Y
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))  

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].X
        y = points.iloc[i].Y
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b  

m = 0  
b = 0 
L = 0.0001 
epochs = 300 

for i in range(epochs):
    m, b = gradient_descent(m, b, data, L)
    if i % 50 == 0:
        print(f"Epoch {i} Loss: {loss_function(m, b, data)}")

print(f"Final values: m = {m}, b = {b}")

plt.scatter(data.X, data.Y, color="red", label="Actual Data")


x_range = np.linspace(data.X.min(), data.X.max(), 100)  
y_range = m * x_range + b  
plt.plot(x_range, y_range, color="black", linewidth=2, label="Regression Line")

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
