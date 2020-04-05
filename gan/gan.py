import sys
sys.path.append("..")
from neural_network.neural_network import NeuralNetwork
from nn_components.losses import BinaryCrossEntropy
from nn_components.layers import InputLayer, FCLayer, ActivationLayer, BatchNormLayer, DropoutLayer, LearnableLayer
from optimizations_algorithms.optimizers import Adam
from libs.utils import TrainerGAN, load_dataset_mnist, preprocess_data
from libs.mnist_lib import MNIST

class Generator(NeuralNetwork):
    
    def __init__(self, optimizer:object, layers:list, loss_func:object=None):
        super().__init__(optimizer, layers, loss_func)
        self.learnable_layers = [layer for layer in self.layers if isinstance(layer, LearnableLayer)][::-1]

    def _backward_last(self):
        pass

    def backward(self, y, y_hat, z, discriminator):
        """
        Generator don't compute directly with loss, so we need discriminator 
            backprop gradient from earlier layers according backward direction.
                    
        Parameters
        ----------
        y: vector ones (for optimizing fake to real), we want optimize generator parameters to fool discriminator.
        y_hat: output pass from generator -> discriminator
        z: random noise variables.
        discriminator: discriminator network. 
        """
        dA = discriminator.return_input_grads(y, y_hat)
        grads = self._backward(dA, None)
        self._update_params(grads)

class Discriminator(NeuralNetwork):

    def __init__(self, optimizer:object, layers:list, loss_func:object=BinaryCrossEntropy()):
        super().__init__(optimizer, layers, loss_func)

    def return_input_grads(self, y, y_hat):
        """
        Compute gradient of Loss w.r.t inputs, flow gradient to compute gradient of Loss w.r.t generator parameters.
        """
        dA_prev, _ = self._backward_last(y, y_hat)
        for i in range(len(self.layers)-1, 0, -1):
            backward_func = self.layers[i].backward_layer if isinstance(self.layers[i], LearnableLayer) else self.layers[i].backward
            dA_prev = backward_func(dA_prev, self.layers[i-1])
        return dA_prev


def main(digit=2):

    mnist_dim = 784

    arch_generator = [
        InputLayer(),

        FCLayer(num_neurons=128, weight_init="he_normal"),
        BatchNormLayer(),
        ActivationLayer(activation="relu"),

        FCLayer(num_neurons=256, weight_init="he_normal"),
        BatchNormLayer(),
        ActivationLayer(activation="relu"),

        FCLayer(num_neurons=512, weight_init="he_normal"),
        BatchNormLayer(),
        ActivationLayer(activation="relu"),

        FCLayer(num_neurons=mnist_dim, weight_init="he_normal"),
        BatchNormLayer(),
        ActivationLayer(activation="tanh"),
    ]

    arch_discriminator = [
        InputLayer(),

        FCLayer(num_neurons=mnist_dim, weight_init="he_normal"),
        ActivationLayer(activation="relu"),
        DropoutLayer(keep_prob=0.8),

        FCLayer(num_neurons=512, weight_init="he_normal"),
        ActivationLayer(activation="relu"),
        DropoutLayer(keep_prob=0.8),

        FCLayer(num_neurons=256, weight_init="he_normal"),
        ActivationLayer(activation="relu"),
        DropoutLayer(keep_prob=0.9),

        FCLayer(num_neurons=128, weight_init="he_normal"),
        ActivationLayer(activation="relu"),

        FCLayer(num_neurons=1, weight_init="he_normal"),
        ActivationLayer(activation="sigmoid"),
    ]

    load_dataset_mnist("../libs")
    mndata = MNIST('../libs/data_mnist', gz=True)

    images, labels = mndata.load_training()
    images, labels = preprocess_data(images, labels, test=True)

    images = images[labels == digit]

    optimizer_G = Adam(alpha=0.006)
    optimizer_D = Adam(alpha=0.006)

    loss_func = BinaryCrossEntropy()

    generator = Generator(optimizer=optimizer_G, layers=arch_generator)
    discriminator = Discriminator(optimizer=optimizer_D, layers=arch_discriminator, loss_func=loss_func)

    batch_size = 64
    iterations = 10000

    print("Training GAN with MNIST dataset to generate digit %d" % digit)
    trainerGAN = TrainerGAN(generator, discriminator, batch_size, iterations)
    trainerGAN.train(images)

if __name__ == "__main__":
    main(digit=8)