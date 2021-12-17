import mnist_loader
import relu_classifier as rel
import sigmoid_classifier as sig

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

af = int(input('1 for sigmoid, 2 for relu, 3 for tanh'))

if af == 1:
    classifier = sig
elif af == 2:
    classifier = rel
else:
    quit()

net = classifier.Network([784,30,10])

net.SGD(training_data, 25, 10, 0.05, test_data=test_data)
