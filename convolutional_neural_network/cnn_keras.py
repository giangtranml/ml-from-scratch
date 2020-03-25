from convolutional_neural_network import CNN
import keras
L = keras.layers
M = keras.models

class CNNKeras(CNN):

    initializer_mapping = {"he": keras.initializers.he_normal(), "std": keras.initializers.RandomNormal(0, stddev=1), 
                           "xavier": keras.initializers.glorot_normal()}

    def _structure(self, cnn_structure):
        """
        Structure function that initializes CNN architecture.

        Parameters
        ----------
        cnn_structure: (list) a list of dictionaries define CNN architecture.
            Each dictionary element is 1 kind of layer (ConvLayer, FCLayer, PoolingLayer, FlattenLayer, BatchNormLayer).

        - Convolutional layer (`type: conv`) dict should have following key-value pair:
            + filter_size: (tuple) define conv filter size (fH, fW)
            + filters: (int) number of conv filters at the layer.
            + stride: (int) stride of conv filter.
            + weight_init: (str) choose which kind to initialize the filter, either `he` `xavier` or `std`.
            + padding: (str) padding type of input corresponding to the output, either `SAME` or `VALID`.
            + activation (optional): (str) apply activation to the output of the layer. LINEAR -> ACTIVATION.
            + batch_norm (optional): (any) apply batch norm to the output of the layer. LINEAR -> BATCH NORM -> ACTIVATION
        
        - Pooling layer (`type: pool`) dict should have following key-value pair:
            + filter_size: (tuple) define pooling filter size (fH, fW).
            + mode: (str) choose the mode of pooling, either `max` or `avg`.
            + stride: (int) stride of pooling filter.

        - Fully-connected layer (`type: fc`) dict should have following key-value pair:
            + num_neurons: (int) define number of neurons in the dense layer.
            + weight_init: (str) choose which kind to initialize the weight, either `he` `xavier` or `std`.
            + activation (optional): (str) apply activation to the output of the layer. LINEAR -> ACTIVATION.
            + batch_norm (optional): (any) apply batch norm to the output of the layer. LINEAR -> BATCH NORM -> ACTIVATION
        
        """
        model = M.Sequential()
        model.add(L.InputLayer(input_shape=(28, 28, 1)))
        for struct in cnn_structure:
            if type(struct) is str and struct == "flatten":
                model.add(L.Flatten())
                continue
            if struct["type"] == "conv":
                filter_size = struct["filter_size"]
                filters = struct["filters"]
                padding = struct["padding"]
                stride = struct["stride"]
                weight_init = struct["weight_init"]
                model.add(L.Conv2D(filters=filters, kernel_size=filter_size, padding=padding, strides=stride, 
                                    kernel_initializer=self.initializer_mapping[weight_init]))
                if "batch_norm" in struct:
                    model.add(L.BatchNormalization())
                if "activation" in struct:
                    activation = struct["activation"]
                    model.add(L.Activation(activation=activation))
            elif struct["type"] == "pool":
                filter_size = struct["filter_size"]
                stride = struct["stride"]
                mode = struct["mode"]
                if mode == "avg":
                    model.add(L.AveragePooling2D(pool_size=filter_size, strides=stride))
                else:
                    model.add(L.MaxPool2D(pool_size=filter_size, strides=stride))
            else:
                num_neurons = struct["num_neurons"]
                weight_init = struct["weight_init"]
                model.add(L.Dense(units=num_neurons, kernel_initializer=self.initializer_mapping[weight_init]))
                if "batch_norm" in struct:
                    model.add(L.BatchNormalization())
                if "activation" in struct:
                    activation = struct["activation"]
                    model.add(L.Activation(activation=activation))
        return model

    def train(self, X_train, Y_train):
        """
        Training function.

        Parameters
        ----------
        X_train: training dataset X.
        Y_train: one-hot encoding label.
        """

        self.layers.compile(optimizer=self.optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        self.layers.fit(X_train, Y_train, batch_size=self.batch_size, epochs=self.epochs)

    def predict(self, test_X):
        """
        Predict function.
        """
        y_hat = self.layers.predict(test_X)
        return np.argmax(y_hat, axis=1)

