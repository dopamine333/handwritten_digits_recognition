import numpy as np
import random
import json
from json import JSONEncoder
import math

from numpy.core.fromnumeric import size

# TODO 載入訓練完成的權重與偏權
# TODO pygame手寫功能
# TODO 訓練與實用分開
# TODO network封裝


class NumpyArrayEncoder(JSONEncoder):
    def default(self, object):
        if isinstance(object, np.ndarray):
            return object.tolist()
        return JSONEncoder.default(self, object)


def reLU(x):
    return x * (x > 0)


def reLU_prime(x):
    return 1 * (x > 0)


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1/(1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))


class NeuralNetwork:
    def __init__(self, layers: list[int], learning_rate: float, mini_batch_size: int, nonlinear_funcs=(sigmoid, sigmoid_prime)) -> None:
        self.layers = layers
        self.layer_num = len(self.layers)-1

        self.weights: list[np.ndarray] = [
            np.random.randn(right, left) for left, right in zip(self.layers[:-1], self.layers[1:])]
        self.biases: list[np.ndarray] = [
            np.random.randn(height, 1) for height in self.layers[1:]]

        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        self.nonlinear_func, self.nonlinear_func_prime = nonlinear_funcs

    def create_neurons(self):
        return [np.zeros((height, 1)) for height in self.layers[1:]]

    def _feedforward(self, inputs: np.ndarray, neurons: list[np.ndarray]):

        neurons[0] = self.nonlinear_func(
            self.weights[0]@inputs+self.biases[0])
        for k in range(1, self.layer_num):
            neurons[k] = self.nonlinear_func(
                self.weights[k]@neurons[k-1]+self.biases[k])

    def _backpropagation(self, inputs: np.ndarray, label: np.ndarray, neurons: list[np.ndarray]) -> tuple[list[np.ndarray]]:
        gradient_neurons: list[np.ndarray] = []
        gradient_weights: list[np.ndarray] = []
        gradient_biases: list[np.ndarray] = []

        # get input or neurons
        def get_neurons(k):
            if k < -self.layer_num:
                return inputs
            return neurons[k]

        # output/rightst layer
        gradient_neurons.append(neurons[-1]-label)

        # hidden/middle layers
        for k in range(1, self.layer_num+1):
            z = self.nonlinear_func_prime(self.weights[-k]@get_neurons(-k-1) +
                                          self.biases[-k])*gradient_neurons[k-1]

            gradient_weights.append(np.transpose(get_neurons(-k-1))*z)  # w
            gradient_biases.append(z)  # b
            if k == self.layer_num:
                break
            gradient_neurons.append(
                np.sum(np.transpose(self.weights[-k]*z), axis=1, keepdims=True))  # x

        gradient_weights.reverse()
        gradient_biases.reverse()

        return (gradient_weights, gradient_biases)

    def _gradient_descent(self, weights_gradients: list[np.ndarray], biases_gradients: list[np.ndarray]):
        for k in range(self.layer_num):
            self.weights[k] -= self.learning_rate*weights_gradients[k]
            self.biases[k] -= self.learning_rate*biases_gradients[k]

    def training(self, inputs_list: list[np.ndarray], labels_list: list[np.ndarray], print_progress=True):

        inputs_list_len = len(inputs_list)
        random_indexs = np.random.permutation(np.arange(inputs_list_len))

        def get_mini_batches(n):
            return [(inputs_list[index], labels_list[index])
                    for index in random_indexs[n:min(n+self.mini_batch_size, inputs_list_len)]]
        neurons=self.create_neurons()

        average_weights_gradients: list[np.ndarray] = [
            np.zeros_like(self.weights[k]) for k in range(len(self.weights))
        ]
        average_biases_gradients: list[np.ndarray] = [
            np.zeros_like(self.biases[k]) for k in range(len(self.biases))
        ]

        mini_batches_num = 0
        if print_progress:
            n = 0

        while True:
            for inputs, label in get_mini_batches(mini_batches_num):
                self._feedforward(inputs, neurons)
                weights_gradients, biases_gradients = \
                    self._backpropagation(inputs, label, neurons)

                for layer in range(self.layer_num):
                    average_weights_gradients[layer] += weights_gradients[layer]
                    average_biases_gradients[layer] += biases_gradients[layer]

            self._gradient_descent(average_weights_gradients,
                                   average_biases_gradients)

            for layer in range(self.layer_num):
                average_weights_gradients[layer].fill(0)
                average_biases_gradients[layer].fill(0)

            mini_batches_num += self.mini_batch_size

            if print_progress:
                n += 1
                print(
                    f"training : {n} / {math.ceil(inputs_list_len/self.mini_batch_size)}")

            if mini_batches_num >= inputs_list_len:
                if print_progress:
                    print("done! training completed")
                return

    def save(self, data_path: str):
        network_data = {
            "learning_rate": self.learning_rate,
            "mini_batch_size": self.mini_batch_size,
            "weights": self.weights,
            "biases": self.biases
        }
        with open(data_path, "w") as write_file:
            json.dump(network_data, write_file,
                      cls=NumpyArrayEncoder, indent=4)
        print(f"Done writing serialized NeuralNetwork data into {data_path} file")

    def load(self, data_path: str):
        with open(data_path, "r") as read_file:
            decoded_network = json.load(read_file)

            self.weights = [np.asarray(w) for w in decoded_network["weights"]]
            self.biases = [np.asarray(b) for b in decoded_network["biases"]]
            self.learning_rate = decoded_network["learning_rate"]
            self.mini_batch_size = decoded_network["mini_batch_size"]

    def create_from_file(data_path: str):
        with open(data_path, "r") as read_file:
            decoded_network = json.load(read_file)
            weights = [np.asarray(w) for w in decoded_network["weights"]]
            biases = [np.asarray(b) for b in decoded_network["biases"]]
            learning_rate = decoded_network["learning_rate"]
            mini_batch_size = decoded_network["mini_batch_size"]

            layers = [w.shape[1] for w in weights]+[weights[-1].shape[0]]
            network = NeuralNetwork(layers, learning_rate, mini_batch_size)
            network.weights = weights
            network.biases = biases

            return network

    def get(self, inputs: np.ndarray) -> np.ndarray:
        neurons=self.create_neurons()
        self._feedforward(inputs, neurons)
        return neurons[-1]

    def info(self):
        return f"""
    layers : {self.layers}
    learning_rate : {self.learning_rate}
    mini_batch_size : {self.mini_batch_size}
    nonlinear_func : {self.nonlinear_func.__name__}
    nonlinear_func_prime : {self.nonlinear_func_prime.__name__}
    
    """
