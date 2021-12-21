import csv
from matplotlib import pyplot as plt
import numpy as np

results_data = []

with open('results.csv', 'r', newline='') as f:
    reader = csv.reader(f)

    for row in reader:
        results_data.append(row)

relu_res = list(filter(lambda x: x[1] == 'relu', results_data))
sigmoid_res = list(filter(lambda x: x[1] == 'sigmoid', results_data))
tanh_res = list(filter(lambda x: x[1] == 'tanh', results_data))

avg_relu_time = sum([float(x[2]) for x in relu_res])/len(relu_res)
avg_relu_acc = sum([float(x[3]) for x in relu_res])/len(relu_res)

avg_sigmoid_time = sum([float(x[2]) for x in sigmoid_res])/len(sigmoid_res)
avg_sigmoid_acc = sum([float(x[3]) for x in sigmoid_res])/len(sigmoid_res)

avg_tanh_time = sum([float(x[2]) for x in tanh_res])/len(tanh_res)
avg_tanh_acc = sum([float(x[3]) for x in tanh_res])/len(tanh_res)


print(f'times: {avg_relu_time} {avg_sigmoid_time} {avg_tanh_time}')
print(f'accuracies: {avg_relu_acc} {avg_sigmoid_acc} {avg_tanh_acc}')

plt.bar(['ReLU','Sigmoid','Hyperbolic Tangent'], [avg_relu_time, avg_sigmoid_time, avg_tanh_time])
plt.yticks(np.arange(0,6,0.5))
plt.title('Average Training Time for Neural Networks Using a Given Activation Funciton')
plt.xlabel('Activation Function')
plt.ylabel('Training Time (s)')
plt.show()

plt.bar(['ReLU','Sigmoid','Hyperbolic Tangent'], [avg_relu_acc, avg_sigmoid_acc, avg_tanh_acc])
plt.yticks(np.arange(0,105,5))
plt.title('Average Accuracy of Neural Networks Using a Given Activation Funciton')
plt.xlabel('Activation Function')
plt.ylabel('Accuracy (%)')
plt.show()