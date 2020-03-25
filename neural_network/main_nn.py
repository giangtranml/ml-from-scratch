import sys
sys.path.append("..")
import os
from optimizations_algorithms.optimizers import SGD, SGDMomentum, RMSProp, Adam
from neural_network import NeuralNetwork
from nn_components.layers import FCLayer, ActivationLayer, DropoutLayer, BatchNormLayer
from nn_components.losses import CrossEntropy
from libs.utils import load_dataset_mnist, preprocess_data, Trainer, Evaluator
from libs.mnist_lib import MNIST


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
        FCLayer(num_neurons=100, weight_init="he_normal"),
        ActivationLayer(activation="relu"),
        DropoutLayer(keep_prob=0.8),

        FCLayer(num_neurons=125, weight_init="he_normal"),
        ActivationLayer(activation="relu"),
        DropoutLayer(keep_prob=0.8),

        FCLayer(num_neurons=50, weight_init="he_normal"),
        ActivationLayer(activation="relu"),
        BatchNormLayer(),

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