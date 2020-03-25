import numpy as np
import keras
L = keras.layers
M = keras.models
from recurrent_neural_network import RecurrentNeuralNetwork

class RNNKeras(RecurrentNeuralNetwork):

   def train(self, X_train, Y_train):
        """
        X_train: shape=(m, time_steps)
        Y_train: shape=(m, time_steps, vocab_length)
        """
        m, time_steps, vocab_length = Y_train.shape
        input_ = L.Input(shape=(time_steps,))
        X = L.Embedding(input_dim=vocab_length, output_dim=16)(input_)
        X = L.SimpleRNN(units=self.hidden_units, return_sequences=True)(X)

        logits = L.TimeDistributed(L.Dense(units=vocab_length, activation="softmax"))(X)

        model = M.Model(inputs=input_, outputs=logits)
        model.summary()

        model.compile(optimizer=self.optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        model.fit(x=X_train, y=Y_train, batch_size=self.batch_size, epochs=self.epochs)