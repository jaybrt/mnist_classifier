import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10, 10, 0.01)

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def relu(x):
    return np.maximum(0,x)


plt.figure(figsize=(9,3))

plt.subplot(1)
plt.title('Sigmoid')
plt.plot(x, sigmoid(x))

plt.subplot(2)
plt.title('Hyperbolic Tangent')
plt.plot(x, tanh(x))

plt.subplot(3)
plt.title('Rectified Linear Unit')
plt.plot(x, relu(x))
plt.show()