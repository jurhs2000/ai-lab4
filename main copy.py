from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

def initialize_parameters(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]
    return parameters

def linear_activation_forward(A_prev, W, b, activation):
    if activation == 'sigmoid':
        Z = np.dot(W, A_prev) + b
        A = 1 / (1 + np.exp(-Z))
        cache = (A_prev, W, b)
    elif activation == 'relu':
        Z = np.dot(W, A_prev) + b
        A = np.maximum(0, Z)
        cache = (A_prev, W, b)
    return A, cache

def compute_cost(AL, y):
    m = y.shape[0]
    cost = -1 / m * np.sum(y * np.log(AL) + (1 - y) * np.log(1 - AL))
    return cost

def relu_backward(dA, activation_cache):
    Z = activation_cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1 / m * np.dot(dZ, A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def sigmoid_backward(dA, activation_cache):
    Z = activation_cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def L_model_backward(AL, y, caches):
    grads = {}
    L = len(caches)
    m = y.shape[1]
    dAL = - (np.divide(y, AL) - np.divide(1 - y, 1 - AL))
    current_cache = caches[L - 1]
    grads['dA' + str(L - 1)], grads['dW' + str(L)], grads['db' + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l + 1)], current_cache, 'relu')
        grads['dA' + str(l)] = dA_prev_temp
        grads['dW' + str(l + 1)] = dW_temp
        grads['db' + str(l + 1)] = db_temp
    return grads

def L_model_forward(x, parameters):
    caches = []
    A = x
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid')
    caches.append(cache)
    return AL, caches

def L_layer_datamodel(x, y, layers_dims, learning_rate, iterations):
    costs = []
    parameters = []
    L = len(layers_dims)
    parameters = initialize_parameters(layers_dims)
    for i in range(iterations):
        AL, caches = L_model_forward(x, parameters)
        cost = compute_cost(AL, y)
        grads = L_model_backward(AL, y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if i % 100 == 0:
            costs.append(cost)
            print("Cost after iteration %i: %f" % (i, cost))
    return parameters, costs

# Creando dataset
x, y = make_classification(n_samples=50, n_features=3, n_informative=3, n_redundant=0, n_classes=2)
print(x)
print(y)

parameters = L_layer_datamodel(x, y, [50, 20, 7, 5, 1], 0.007, 100)

# predict
