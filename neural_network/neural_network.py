"""
Author: Giang Tran
Email: giangtran240896@gmail.com
Docs: https://giangtranml.github.io/ml/neural-network.html
"""
import sys, os
sys.path.append("..")
import numpy as np
from nn_components.layers import FCLayer, ActivationLayer, BatchNormLayer, DropoutLayer, LearnableLayer, InputLayer
from nn_components.losses import CrossEntropy
from optimizations_algorithms.optimizers import Adam, SGD, SGDMomentum, RMSProp
from libs.utils import load_dataset_mnist, preprocess_data, Trainer, Evaluator
from libs.mnist_lib import MNIST

class NeuralNetwork:

    def __init__(self, optimizer:object, layers:list, loss_func:object=CrossEntropy()):
        """
        Deep neural network architecture.

        Parameters
        ----------
        optimizer: (object) optimizer object uses to optimize the loss.
        layers: (list) a list of sequential layers. For neural network, it should have [FCLayer, ActivationLayer, BatchnormLayer, DropoutLayer]
        loss_func: (object) the type of loss function we want to optimize. 
        """
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.layers = layers

    def _forward(self, train_X, prediction=False):
        """
        NN forward propagation level.

        Parameters
        ----------
        train_X: training dataset X.
                shape = (N, D)
        prediction: whether this forward pass is prediction stage or training stage.

        Returns
        -------
        Probability distribution of softmax at the last layer.
            shape = (N, C)
        """
        inputs = train_X
        layers = self.layers
        if hasattr(self, "output_layers"):
            layers = layers + self.output_layers
        for layer in layers:
            if isinstance(layer, (BatchNormLayer, DropoutLayer)):
                inputs = layer.forward(inputs, prediction=prediction)
                continue
            inputs = layer.forward(inputs)
        output = inputs
        return output

    def _backward_last(self, Y, Y_hat):
        """
        Special formula of backpropagation for the last layer.
        """
        if not hasattr(self, "output_layers"):
            self.output_layers = self.layers[-2:]
            self.layers = self.layers[:-2]
            self.learnable_layers = [layer for layer in self.layers if isinstance(layer, LearnableLayer)]
            self.learnable_layers.extend(layer for layer in self.output_layers if isinstance(layer, LearnableLayer))
            self.learnable_layers = self.learnable_layers[::-1]

        delta = self.loss_func.backward(Y_hat, Y)
        dW_last = self.layers[-1].output.T.dot(delta)
        dA_last = delta.dot(self.output_layers[0].W.T)
        return dA_last, dW_last

    def _backward(self, dA_last, dW_last):
        """
        NN backward propagation level. Update weights of the neural network.

        """
        dA_prev, dW = dA_last, dW_last
        grads = [dW]
        if dW is None:
            grads.pop()
        for i in range(len(self.layers)-1, 0, -1):
            if isinstance(self.layers[i], LearnableLayer):
                dA_prev, dW = self.layers[i].backward(dA_prev, self.layers[i-1])
                grads.append(dW)
                continue
            dA_prev = self.layers[i].backward(dA_prev, self.layers[i-1])
        return grads
    
    def _update_params(self, grads):
        self.optimizer.step(grads, self.learnable_layers )

    def backward(self, Y, Y_hat, X):
        """

        Parameters
        ----------
        Y: one-hot encoding label.
            shape = (N, C).
        Y_hat: output values of forward propagation NN.
            shape = (N, C).
        X: training dataset.
            shape = (N, D).
        """
        dA_last, dW_last = self._backward_last(Y, Y_hat)
        grads = self._backward(dA_last, dW_last)
        self._update_params(grads)

    def __call__(self, X, prediction=False):
        return self._forward(X, prediction)

    def predict(self, test_X):
        """
        Predict function.
        """
        y_hat = self._forward(test_X, prediction=True)
        return np.argmax(y_hat, axis=1)


def main():
    load_dataset_mnist("../libs")
    mndata = MNIST('../libs/data_mnist', gz=True)
    weight_path = "nn_weights.pkl"
    training_phase = weight_path not in os.listdir(".")
    if training_phase:
        images, labels = mndata.load_training()
        images, labels = preprocess_data(images, labels)
        epochs = 10
        batch_size = 64
        learning_rate = 0.01

        optimizer = Adam(learning_rate)
        loss_func = CrossEntropy()
        archs = [
            InputLayer(),

            FCLayer(num_neurons=100, weight_init="he_normal"),
            ActivationLayer(activation="relu"),
            DropoutLayer(keep_prob=0.8),

            FCLayer(num_neurons=125, weight_init="he_normal"),
            ActivationLayer(activation="relu"),
            DropoutLayer(keep_prob=0.8),

            FCLayer(num_neurons=50, weight_init="he_normal"),
            BatchNormLayer(),
            ActivationLayer(activation="relu"),

            FCLayer(num_neurons=labels.shape[1], weight_init="he_normal"),
            ActivationLayer(activation="softmax"),
        ]
        nn = NeuralNetwork(optimizer=optimizer, layers=archs, loss_func=loss_func)

        trainer = Trainer(nn, batch_size, epochs)
        trainer.train(images, labels)
        trainer.save_model("nn_weights.pkl")
    else:
        import pickle
        images_test, labels_test = mndata.load_testing()
        images_test, labels_test = preprocess_data(images_test, labels_test, test=True)
        with open(weight_path, "rb") as f:
            nn = pickle.load(f)
        pred = nn.predict(images_test)

        print("Accuracy:", len(pred[labels_test == pred]) / len(pred))
        from sklearn.metrics.classification import confusion_matrix

        print("Confusion matrix: ")
        print(confusion_matrix(labels_test, pred))

if __name__ == "__main__":
    main()