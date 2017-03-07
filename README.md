# digits_recognition

```{.python .input  n=82}
import random
import numpy as np

import _pickle as cPickle
import gzip
import matplotlib.pyplot as plt
```

```{.python .input  n=81}
A = [x for x in range(10)]
print(A)
B = [x for x in range(0,-10, -1)]
print(B)
print(list(zip(A,B)))
```

```{.json .output n=81}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]\n[0, -1, -2, -3, -4, -5, -6, -7, -8, -9]\n[(0, 0), (1, -1), (2, -2), (3, -3), (4, -4), (5, -5), (6, -6), (7, -7), (8, -8), (9, -9)]\n"
 }
]
```

```{.python .input  n=62}
# Input neural network
sizes = [784, 30, 10]
num_layers = len(sizes)
biases = [np.random.randn(y, 1) for y in sizes[1:]]
weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
```

```{.python .input  n=63}
# Load data set
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

f = gzip.open('mnist.pkl.gz', 'rb')
tr_d, va_d, te_d = cPickle.load(f, encoding='latin1')
f.close()
training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
training_results = [vectorized_result(y) for y in tr_d[1]]
training_data = zip(training_inputs, training_results)
validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
validation_data = zip(validation_inputs, va_d[1])
test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
test_data = zip(test_inputs, te_d[1])
```

```{.python .input  n=64}
#net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
#def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):

def cost_derivative(output_activations, y):
    """Return the vector of partial derivatives \partial C_x /
    \partial a for the output activations."""
    return (output_activations-y)

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def feedforward(a):
    """Return the output of the network if "a" is input. """
    for b, w in zip(biases, weights):
        a = sigmoid(np.dot(w, a) + b)
    return a

def backprop(x, y):
    """Return a tuple ``(nabla_b, nabla_w)`` representing the
    gradient for the cost function C_x.  ``nabla_b`` and
    ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
    to ``self.biases`` and ``self.weights``."""
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_w = [np.zeros(w.shape) for w in weights]
    # feedforward
    activation = x
    activations = [x] # list to store all the activations, layer by layer
    zs = [] # list to store all the z vectors, layer by layer
    for b, w in zip(biases, weights):
        z = np.dot(w, activation)+b
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)
    # backward pass
    delta = cost_derivative(activations[-1], y) * \
        sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    # Note that the variable l in the loop below is used a little
    # differently to the notation in Chapter 2 of the book.  Here,
    # l = 1 means the last layer of neurons, l = 2 is the
    # second-last layer, and so on.  It's a renumbering of the
    # scheme in the book, used here to take advantage of the fact
    # that Python can use negative indices in lists.
    for l in range(2, num_layers):
        z = zs[-l]
        sp = sigmoid_prime(z)
        delta = np.dot(weights[-l+1].transpose(), delta) * sp
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
    return (nabla_b, nabla_w)

def evaluate(test_data):
    """Return the number of test inputs for which the neural
    network outputs the correct result. Note that the neural
    network's output is assumed to be the index of whichever
    neuron in the final layer has the highest activation."""
    test_results = [(np.argmax(feedforward(x)), y)
                    for (x, y) in test_data]
    return sum(int(x == y) for (x, y) in test_results)

def update_mini_batch( mini_batch, eta):
    global biases
    global weights
    """Update the network's weights and biases by applying
    gradient descent using backpropagation to a single mini batch.
    The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
    is the learning rate."""
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_w = [np.zeros(w.shape) for w in weights]
    for x, y in mini_batch:
        delta_nabla_b, delta_nabla_w = backprop(x, y)
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    weights = [w-(eta/len(mini_batch))*nw
                    for w, nw in zip(weights, nabla_w)]
    biases = [b-(eta/len(mini_batch))*nb
                   for b, nb in zip(biases, nabla_b)]
```

```{.python .input  n=95}
epochs = 30
mini_batch_size = 10
training_data = list(training_data)
n = len(training_data)
eta = 3.0

if test_data:
    test_data = list(test_data)
    n_test = len(test_data)
    
print("len(training_data) =", n)
```

```{.json .output n=95}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "len(training_data) = 50000\n"
 }
]
```

```{.python .input  n=96}
epochs_it = [0] * epochs
epochs_eval = [0] * epochs
epochs_nTest = [0] * epochs

for j in range(epochs):
    random.shuffle(training_data)
    mini_batches = [
        training_data[k:k+mini_batch_size]
        for k in range(0, n, mini_batch_size)]
    for mini_batch in mini_batches:
        update_mini_batch(mini_batch, eta)
    if test_data:
        epochs_it[j] = j
        epochs_eval[j] = evaluate(test_data)
        epochs_nTest[j] = n_test
        print("Epoch {} : {} / {}".format(epochs_it[j],epochs_eval[j],epochs_nTest[j]));
    else:
        print("Epoch {} complete".format(j))
```

```{.json .output n=96}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0 : 9518 / 10000\nEpoch 1 : 9522 / 10000\nEpoch 2 : 9536 / 10000\nEpoch 3 : 9529 / 10000\nEpoch 4 : 9513 / 10000\nEpoch 5 : 9522 / 10000\nEpoch 6 : 9513 / 10000\nEpoch 7 : 9528 / 10000\nEpoch 8 : 9543 / 10000\nEpoch 9 : 9522 / 10000\n"
 }
]
```

```{.python .input  n=100}
plt.plot(epochs_it, epochs_eval, color='darkorange', label='Y')
plt.hold('on')
plt.xlabel('iteration')
plt.ylabel('correct evaluated')
plt.title('Results')
plt.legend()
plt.show()
```

```{.json .output n=100}
[
 {
  "data": {
   "image/png": "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\njJb8kjDtMwsgtQB2Za1rAix19yWpn48ADieMtHxEWHNyA/And19fT2ibAIYMGUJxcXGG3m28unTp\noveSUHo/yZVP7wX0fpIsn95LDU1eFpGI5ITQdOvnhK6la4DpwNXuvt3MtgMHAucSuo4uIyQl19S1\nTqSG2otpNgNnERKdAsLi2t8QFtKKiIhIQiQiOXH3hwgNwep6bhMwNsI1W9f6eS6hAkhEREQSLAn7\nnIiIiIh8QslJCzNu3Li4Q8iYfHovoPeTZPn0XkDvJ8ny6b2kIxH7nCSVmRUDr7766qv5uEBJREQa\n6Z133mH16tVxh5E43bt3p2/fvnU+V1payogRIyDsWVbalOsmYs2JiIhIUr3zzjsMGTKEjRs3xh1K\n4nTo0IH58+fXm6BEpeRERERkF1avXs3GjRvzas+rTKjex2T16tVKTkREROKQT3teJZ0WxIqIiEii\nKDkRERGRRFFyIiIiIomi5EREREQSRcmJiIiIJIqSExERkRbqpJNOolu3bqxatWqn5z788EP69OnD\nyJHN35ZOyYmIiEgLddttt7FlyxYmTpy403Pf//73WbNmDXfeeWezx6XkREREpIUqKiri2muvZdq0\naTz11FOfHH/55Ze54447uOyyyxg+fHizx6XkREREpAX77ne/ywEHHMAll1zCli1bqKqq4pvf/Cb9\n+/fnmmuuiSUm7RArIiLSgrVu3Zo//OEPHHnkkfzkJz+hR48evPbaa8yaNYt27drFEpOSExERkRbu\nM5/5DJdccgm/+tWvKCgo4Oyzz+Zzn/tcbPEoOREREcmUrRthTUX279NtMLTtkNFL/uxnP2P69OlU\nVlZyww03ZPTaTaXkREREJFPWVMB9I7J/n/GvQq/MNiHcfffd2X///fnggw/o0aNHRq/dVEpORERE\nMqXb4JA4NMd98piSExERkUxp2yHjIxotkUqJRUREJFGUnIiIiEiiKDkRERGRT5hZ3CFozYmIiIgE\ns2fPjjsEQCMnIiIikjBKTkRERCRRlJyIiIhIoig5ERERkURRciIiIiKJouREREREEkXJiYiIiCSK\nkhMRERFJlERswmZmnYCfAqcBPYFS4Dvu/krq+buB82q97O/uflI913scKAFOc/e/1jjeFfgdcDJQ\nBTwMXOruH2f2HYmISL6ZP39+3CEkSjZ/H4lIToDJwFDgHGA5MAF4ysyGuPvy1DmPA+cD1fvqbq7r\n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\nQmCC\n",
   "text/plain": "<matplotlib.figure.Figure at 0x7fe118504a58>"
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
```

```{.python .input}

```
