import numpy as np
from nn_components.initializers import he_normal, xavier_normal, standard_normal, he_uniform, xavier_uniform
from nn_components.activations import relu, sigmoid, tanh, softmax, relu_grad, sigmoid_grad, tanh_grad
import copy
initialization_mapping = {"he_normal": he_normal, "xavier_normal": xavier_normal, "std": standard_normal,
                          "he_uniform": he_uniform, "xavier_uniform": xavier_uniform}


class Layer:

    def forward(self, X):
        raise NotImplementedError("Child class must implement forward() function")

    def backward(self):
        raise NotImplementedError("Child class must implement backward() function")


class LearnableLayer:

    def forward(self, X):
        raise NotImplementedError("Child class must implement forward() function")

    def backward_layer(self):
        pass

    def backward(self):
        raise NotImplementedError("Child class must implement backward() function")

    def update_params(self, grad):
        self.W = self.W - grad

def _split_X(X, filter_size, stride):
    """
    Preprocess input X to avoid for-loop.
    """
    m, iH, iW, iC = X.shape
    fH, fW = filter_size
    oH = int((iH - fH)/stride + 1)
    oW = int((iW - fW)/stride + 1)
    batch_strides, width_strides, height_strides, channel_strides = X.strides
    view_shape = (m, oH, oW, fH, fW, iC)
    X = np.lib.stride_tricks.as_strided(X, shape=view_shape, strides=(batch_strides, stride*width_strides, 
                                                                        stride*height_strides, width_strides, 
                                                                        height_strides, channel_strides), writeable=False)
    return X

class InputLayer(Layer):

    def __init__(self, return_dX=False):
        self.return_dX = return_dX
        self.output = None

    def forward(self, X):
        self.output = X
        return self.output

    def backward(self, d_prev, weights_prev):
        """
        d_prev: gradient of J respect to A[l+1] of the previous layer according backward direction.
        weights_prev: the weights of previous layer according backward direction.
        """
        if self.return_dX:
            return d_prev.dot(weights_prev.T)
        return None


class FCLayer(LearnableLayer):

    def __init__(self, num_neurons, weight_init="std"):
        """
        The fully connected layer.

        Parameters
        ----------
        num_neurons: (integer) number of neurons in the layer.     
        weight_init: (string) either `he_normal`, `xavier_normal`, `he_uniform`, `xavier_uniform` or standard normal distribution.
        """
        assert weight_init in ["std", "he_normal", "xavier_normal", "he_uniform", "xavier_uniform"],\
                "Unknow weight initialization type."
        self.num_neurons = num_neurons
        self.weight_init = weight_init
        self.output = None
        self.W = None

    def forward(self, inputs):
        """
        Layer forward level. 

        Parameters
        ----------
        inputs: inputs of the current layer. This is equivalent to the output of the previous layer.

        Returns
        -------
        output: Output value LINEAR of the current layer.
        """
        if self.W is None:
            self.W = initialization_mapping[self.weight_init](weight_shape=(inputs.shape[1], self.num_neurons))
        self.output = inputs.dot(self.W)
        return self.output

    def backward_layer(self, d_prev, _):
        """
        Compute gradient w.r.t X only.
        """
        d_prev = d_prev.dot(self.W.T)
        return d_prev

    def backward(self, d_prev, prev_layer):
        """
        Layer backward level. Compute gradient respect to W and update it.
        Also compute gradient respect to X for computing gradient of previous
        layers as the forward direction [l-1].

        Parameters
        ----------
        d_prev: gradient of J respect to A[l+1] of the previous layer according backward direction.
        prev_layer: previous layer according forward direction.

        Returns
        -------
        d_prev: gradient of J respect to A[l] at the current layer.
        """
        dW = prev_layer.output.T.dot(d_prev)
        d_prev = self.backward_layer(d_prev, None)
        return d_prev, dW


class ConvLayer(LearnableLayer):

    def __init__(self, filter_size, filters, padding='SAME', stride=1, weight_init="std"):
        """
        The convolutional layer.

        Parameters
        ----------
        filter_size: a 2-elements tuple (width `fH`, height `fW`) of the filter. 
        filters: an integer specifies number of filter in the layer.
        padding: use padding to keep output dimension = input dimension .
                    either 'SAME' or 'VALID'.
        stride: stride of the filters.
        weight_init: (string) either `he_normal`, `xavier_normal`, `he_uniform`, `xavier_uniform` or standard normal distribution.
        """
        assert len(filter_size) == 2, "Filter size must be a 2-elements tuple (width, height)."
        assert weight_init in ["std", "he_normal", "xavier_normal", "he_uniform", "xavier_uniform"],\
                     "Unknow weight initialization type."
        self.filter_size = filter_size
        self.filters = filters
        self.padding = padding
        self.stride = stride
        self.weight_init = weight_init
        self.W = None
        self.output = None

    def _conv_op(self, input_, kernel):
        """
        Convolutional operation of 2 slices.

        Parameters
        ----------
        input_: Input, shape = (m, oH, oW, fH, fW, in_filters)
        kernel: Kernel shape = (fH, fW, in_filters, out_filters)

        Returns
        -------
        Output shape = (m, oH, oW, out_filters)
        """
        return np.einsum("bwhijk,ijkl->bwhl", input_, kernel)

    def _conv_op_backward(self, input_, d_prev, update_params=True):
        """
        Convolutional backward operation.

        Parameters
        ----------
        if update_params is true:
            input_: Input, shape = (m, oH, oW, fH, fW, in_filters)
        else:
            input_: Kernel, shape = (fH, fW, in_filters, out_filters)
        d_prev: Derivative of previous layer. shape = (m, oH, oW, out_filters)

        Returns
        -------
        if update_params is true:
            Derivative with respect to W, shape = (fH, fW, in_filters, out_filters)
        else:
            Derivative with respect to X, shape = (m, oH, oW, fH, fW, in_filters)
        """
        operation = "bwhijk,bwhl->ijkl" if update_params else "ijkl,bwhl->bwhijk"
        return np.einsum(operation, input_, d_prev)

    def _pad_input(self, inp):
        """
        Pad the input when using padding mode 'SAME'.
        """
        m, iH, iW, iC = inp.shape
        fH, fW = self.filter_size
        oH, oW = iH, iW
        pH = int(((oH - 1)*self.stride + fH - iH)/2)
        pW = int(((oW - 1)*self.stride + fW - iW)/2)
        X = np.pad(inp, ((0, 0), (pH, pW), (pH, pW), (0, 0)), 'constant')
        return X

    def forward(self, X):
        """
        Forward propagation of the convolutional layer.

        If padding is 'SAME', we must solve this equation to find appropriate number p:
            oH = (iH - fH + 2p)/s + 1
            oW = (iW - fW + 2p)/s + 1

        Parameters
        ----------
        X: the input to this layer. shape = (m, iH, iW, iC)

        Returns
        -------
        Output value of the layer. shape = (m, oH, oW, filters)
        """
        assert len(X.shape) == 4, "The shape of input image must be a 4-elements tuple (batch_size, height, width, channel)."
        if self.W is None:
            self.W = initialization_mapping[self.weight_init](weight_shape=self.filter_size + (X.shape[-1], self.filters))
        if self.padding == "SAME":
            X = self._pad_input(X)
        X = _split_X(X, self.filter_size, self.stride)
        self.output = self._conv_op(X, self.W)
        return self.output

    def backward_layer(self, d_prev, X):
        _, iH, iW, _ = X.shape
        m, oH, oW, oC = d_prev.shape
        fH, fW = self.filter_size
        dA = np.zeros(shape=(X.shape))
        dA_temp = self._conv_op_backward(self.W, d_prev, update_params=False)
        for h in range(oH):
            for w in range(oW):
                h_step = h*self.stride
                w_step = w*self.stride
                dA[:,  h_step:h_step+fH, w_step:w_step+fW, :] += dA_temp[:, h, w, :, :, :]
        if self.padding == "SAME":
            offset_h = (iH - oH)//2
            offset_w = (iW - oW)//2 
            dA = dA[:, offset_h:-offset_h, offset_w:-offset_w, :]
        return dA

    def backward(self, d_prev, prev_layer):
        """
        Backward propagation of the convolutional layer.
        
        Parameters
        ----------
        d_prev: gradient of J respect to A[l+1] of the previous layer according backward direction.
        prev_layer: previous layer according forward direction.
        
        """
        X = prev_layer.output
        if self.padding == "SAME":
            X = self._pad_input(X)
        dA = self.backward_layer(d_prev, X)
        X = _split_X(X, self.filter_size, self.stride)
        dW = self._conv_op_backward(X, d_prev, update_params=True)
        return dA, dW


class PoolingLayer(Layer):

    def __init__(self, filter_size=(2, 2), stride=2, mode="max"):
        """
        The pooling layer.

        Parameters
        ----------
        filter_size: a 2-elements tuple (width `fH`, height `fW`) of the filter. 
        stride: strides of the filter.
        mode: either average pooling or max pooling.
        """
        assert len(filter_size) == 2, "Pooling filter size must be a 2-elements tuple (width, height)."
        assert mode in ["max", "avg"], "Mode of pooling is either max pooling or average pooling."
        self.filter_size = filter_size
        self.stride = stride
        self.mode = mode

    def _pool_op(self, input_):
        """
        Pooling operation, either max pooling or average pooling.
        
        Parameters
        ----------
        input_: tensor to be pooling, shape = (m, oH, oW, fH, fW, iC).

        Returns
        -------
        Output of the pooling layer, shape = (m, oH, oW, iC).
        """
        if self.mode == "max":
            return np.max(input_, axis=(3, 4))
        else:
            return np.average(input_, axis=(3, 4))

    def _pool_op_backward(self, X, output, d_prev):
        """
        Pooling backpropagation operation. We expect to distribute `d_prev` to appropriate place in the `input_`.

        Parameters
        ----------
        X: input of the pooling layer. shape = (m, oH, oW, fH, fW, out_filters).
        output: output of the pooling layer. shape = (m, oH, oW, out_filters).
        d_prev: derivative of the previous layer according backward direction `l+1`. shape = (m, oH, oW, out_filters).

        Returns
        -------
        Derivative of J respect to this pooling layer `l`. The shape out this gradient will equal the shape of prev_layer output
                                                            with corresponding pooling type (max or avg).
        """
        m, iH, iW, iC = X.shape
        X = _split_X(X, self.filter_size, self.stride)
        m, oH, oW, _ = output.shape
        output = np.reshape(output, newshape=(m, oH, oW, 1, 1, iC))
        d_prev = np.reshape(d_prev, newshape=(m, oH, oW, 1, 1, iC))
        dA = d_prev*(X == output)
        m, oH, oW, fH, fW, _ = dA.shape
        dA = dA.transpose(0, 1, 3, 2, 4, 5).reshape((m, oH*fH, oW*fW, iC))
        if iH - oH*fH > 0 or iW - oW*fW > 0:
            dA = np.pad(dA, ((0, 0), (0, iH - oH*fH), (0, iW - oW*fW), (0, 0)), 'constant')
        return dA

    def forward(self, X):
        """
        Pooling layer forward propagation. Through this layer, the input dimension will reduce:
            oH = floor((iH - fH)/stride + 1)
            oW = floor((iW - fW)/stride + 1)

        Paramters
        ---------
        X: input tensor to this pooling layer. shape=(m, iH, iW, iC)

        Returns
        -------
        Output tensor that has shape = (m, oH, oW, iC)
        """
        X = _split_X(X, self.filter_size, self.stride)
        self.output = self._pool_op(X)
        return self.output

    def backward(self, d_prev, prev_layer):
        """
        Pooling layer backward propagation.

        Parameters
        ----------
        d_prev: gradient of J respect to A[l+1] of the previous layer according backward direction.
        prev_layer: previous layer according forward direction `l-1`.

        Returns
        -------
        Gradient of J respect to this pooling layer `l`. The shape out this gradient will equal the shape of prev_layer output
                                                            with corresponding pooling type (max or avg).
        E.g:
            prev_layer output: [[1, 2],         then max: [[0, 0],      or avg: [[1/4, 2/4],
                                [3, 4]]                    [0, 4]]               [3/4, 4/4]]
        """
        X = prev_layer.output
        d_prev = self._pool_op_backward(X, self.output, d_prev)
        return d_prev


class FlattenLayer(Layer):

    def __init__(self):
        pass

    def forward(self, X):
        """
        Flatten tensor `X` to a vector.
        """
        m, iH, iW, iC = X.shape
        self.output = np.reshape(X, (m, iH*iW*iC))
        return self.output

    def backward(self, d_prev, prev_layer):
        """
        Reshape d_prev shape to prev_layer output shape in the backpropagation.
        """
        m, iH, iW, iC = prev_layer.output.shape
        d_prev = np.reshape(d_prev, (m, iH, iW, iC))
        return d_prev


class ActivationLayer(Layer):

    def __init__(self, activation):
        """
        activation: (string) available activation functions. Must be in [sigmoid, tanh,
                                relu, softmax]. Softmax activation must be at the last layer.
        
        """
        assert activation in ["sigmoid", "tanh", "relu", "softmax"], "Unknown activation function: " + str(activation)
        self.activation = activation

    def forward(self, X):
        """
        Activation layer forward propgation.
        """
        self.output = eval(self.activation)(X)
        return self.output

    def backward(self, d_prev, _):
        """
        Activation layer backward propagation.

        Parameters
        ---------- 
        d_prev: gradient of J respect to A[l+1] of the previous layer according backward direction.
        
        Returns
        -------
        Gradient of J respect to type of activations (sigmoid, tanh, relu) in this layer `l`.
        """
        d_prev = d_prev * eval(self.activation + "_grad")(self.output)
        return d_prev


class DropoutLayer(Layer):

    """
    Refer to the paper: 
        http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
    """

    def __init__(self, keep_prob):
        """
        keep_prob: (float) probability to keep neurons in network, use for dropout technique.
        """
        assert 0.0 < keep_prob < 1.0, "keep_prob must be in range [0, 1]."
        self.keep_prob = keep_prob

    def forward(self, X, prediction=False):
        """
        Drop neurons random uniformly.
        """
        self.mask = np.random.uniform(size=X.shape) < self.keep_prob
        self.output = X * self.mask
        return self.output

    def backward(self, d_prev, _):
        """
        Flow gradient of previous layer [l+1] according backward direction through dropout layer.
        """
        return d_prev * self.mask


class BatchNormLayer(LearnableLayer):

    def __init__(self, momentum=0.99, epsilon=1e-9):
        self.momentum = momentum
        self.epsilon = epsilon
        self.W = None

    def forward(self, X, prediction=False):
        """
        Compute batch norm forward.
        LINEAR -> BATCH NORM -> ACTIVATION.

        Returns
        -------
        Output values of batch normalization.
        """
        if self.W is None:
            gamma = np.ones(((1,) + X.shape[1:]))
            beta = np.zeros(((1,) + X.shape[1:]))
            self.W = np.vstack((gamma, beta))
            self.moving_average = np.zeros_like(self.W) 
        if not prediction:
            self.mu = np.mean(X, axis=0, keepdims=True)
            self.sigma = np.std(X, axis=0, keepdims=True)
            self.moving_average[0] = self.momentum*(self.moving_average[0]) + (1-self.momentum)*self.mu
            self.moving_average[1] = self.momentum*(self.moving_average[1]) + (1-self.momentum)*self.sigma
        else:
            self.mu = self.moving_average[0]
            self.sigma = self.moving_average[1]
        self.Xnorm = (X - self.mu)/np.sqrt(self.sigma + self.epsilon)
        self.output = self.W[0]*self.Xnorm + self.W[1]
        return self.output

    def backward_layer(self, d_prev, prev_layer):
        """
        Compute gradient w.r.t X only.
        
        Parameters
        ---------- 
        d_prev: gradient of J respect to A[l+1] of the previous layer according backward direction.
        prev_layer: previous layer according forward direction.
        """
        m = prev_layer.output.shape[0]
        dXnorm = d_prev * self.W[0]
        dSigma = np.sum(dXnorm * (-((prev_layer.output - self.mu)*(self.sigma+self.epsilon)**(-3/2))/2),
                       axis=0, keepdims=True)
        dMu = np.sum(dXnorm*(-1/np.sqrt(self.sigma+self.epsilon)), axis=0, keepdims=True) +\
                dSigma*((-2/m)*np.sum(prev_layer.output - self.mu, axis=0, keepdims=True))
        d_prev = dXnorm*(1/np.sqrt(self.sigma+self.epsilon)) + dMu/m +\
                dSigma*((2/m)*np.sum(prev_layer.output - self.mu, axis=0, keepdims=True))
        return d_prev

    def backward(self, d_prev, prev_layer):
        """
        Compute batch norm backward.
        LINEAR <- BATCH NORM <- ACTIVATION.
        https://giangtranml.github.io/ml/machine-learning/batch-normalization

        Parameters
        ---------- 
        d_prev: gradient of J respect to A[l+1] of the previous layer according backward direction.
        prev_layer: previous layer according forward direction.
        
        Returns
        -------
        dZ: Gradient w.r.t LINEAR function Z.
        """
        gamma_grad = np.sum(d_prev * self.Xnorm, axis=0, keepdims=True)
        beta_grad = np.sum(d_prev, axis=0, keepdims=True)
        d_prev = self.backward_layer(d_prev, prev_layer)
        return d_prev, np.vstack((gamma_grad, beta_grad))