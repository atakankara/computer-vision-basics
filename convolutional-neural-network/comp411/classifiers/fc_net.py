from builtins import range
from builtins import object
import numpy as np

from comp411.layers import *
from comp411.layer_utils import *


class FourLayerNet(object):
    """
    A four-layer fully-connected neural network with Leaky ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of tuple of (H1, H2, H3) yielding the dimension for the
    first, second and third hidden layer respectively, and perform classification over C classes.

    The architecture should be affine - leakyrelu - affine - leakyrelu - affine - leakyrelu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=(64, 32, 32), num_classes=10,
                 weight_scale=1e-2, reg=5e-3, alpha=1e-3):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: A tuple giving the size of the first, second and third hidden layer respectively
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        - alpha: negative slope of Leaky ReLU layers
        """
        self.params = {}
        self.reg = reg
        self.alpha = alpha

        ############################################################################
        # TODO: Initialize the weights and biases of the four-layer net. Weights   #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1', second layer                    #
        # weights and biases using the keys 'W2' and 'b2', third layer weights and #
        # biases using the keys 'W3' and 'b3' and fourth layer weights and biases  #
        # using keys 'W4' and 'b4'                                                 #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params["W1"] = np.random.normal(0, weight_scale, (input_dim, hidden_dim[0]))
        self.params["b1"] = np.zeros(hidden_dim[0])

        self.params["W2"] = np.random.normal(0, weight_scale, (hidden_dim[0], hidden_dim[1]))
        self.params["b2"] = np.zeros(hidden_dim[1])

        self.params["W3"] = np.random.normal(0, weight_scale, (hidden_dim[1], hidden_dim[2]))
        self.params["b3"] = np.zeros(hidden_dim[2])
    
        self.params["W4"] = np.random.normal(0, weight_scale, (hidden_dim[2], num_classes))
        self.params["b4"] = np.zeros(num_classes)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the four-layer net, computing the   #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        cache_dict = {}

        out, cache = affine_lrelu_forward(X, self.params["W1"], self.params["b1"], {"alpha": self.alpha})
        cache_dict["layer1"] = cache

        out, cache = affine_lrelu_forward(out, self.params["W2"], self.params["b2"], {"alpha": self.alpha})
        cache_dict["layer2"] = cache

        out, cache = affine_lrelu_forward(out, self.params["W3"], self.params["b3"], {"alpha": self.alpha})
        cache_dict["layer3"] = cache

        scores, cache = affine_forward(out, self.params["W4"], self.params["b4"])
        cache_dict["aff4"] = cache
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the four-layer net. Store the loss#
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        l2_reg = np.sum(self.params["W1"]**2) + np.sum(self.params["W2"]**2) \
                      + np.sum(self.params["W3"]**2) + np.sum(self.params["W4"]**2)
        
        loss, dout = softmax_loss(scores, y)
        loss = loss + (0.5 * self.reg * l2_reg)
        
        
        dout, grads["W4"], grads["b4"] = affine_backward(dout, cache_dict["aff4"])
        grads["W4"] += self.reg * self.params["W4"]

        dout, grads["W3"], grads["b3"] = affine_lrelu_backward(dout, cache_dict["layer3"])
        grads["W3"] += self.reg * self.params["W3"]

        dout, grads["W2"], grads["b2"] = affine_lrelu_backward(dout, cache_dict["layer2"])
        grads["W2"] += self.reg * self.params["W2"]

        dout, grads["W1"], grads["b1"] = affine_lrelu_backward(dout, cache_dict["layer1"])
        grads["W1"] += self.reg * self.params["W1"]
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    LeakyReLU nonlinearities, and a softmax loss function. This will also implement
    dropout optionally. For a network with L layers, the architecture will be

    {affine - leakyrelu - [dropout]} x (L - 1) - affine - softmax

    where dropout is optional, and the {...} block is repeated L - 1 times.

    Similar to the FourLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, reg=0.0, alpha=1e-2,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - reg: Scalar giving L2 regularization strength.
        - alpha: negative slope of Leaky ReLU layers
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deterministic so we can gradient check the
          model.
        """
        self.use_dropout = dropout != 1
        self.reg = reg
        self.alpha = alpha
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        prev_dim = input_dim

        for i, current_dim in enumerate(hidden_dims, 1):
            self.params[f"W{i}"] = np.random.normal(0, weight_scale, (prev_dim, current_dim))
            prev_dim = current_dim
            self.params[f"b{i}"] = np.zeros(current_dim)

        i += 1
        self.params[f"W{i}"] = np.random.normal(0, weight_scale, (prev_dim, num_classes))
        self.params[f"b{i}"] = np.zeros(num_classes)

        self.last_index = i

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as FourLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for dropout param since it
        # behaves differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        cache_dict = {}
        out = X
        for i in range(1, self.num_layers+1):
            if i == self.last_index:
                scores, cache = affine_forward(out, self.params[f"W{i}"], self.params[f"b{i}"])
                cache_dict[f"layer{i}"] = cache
            else: 
                out, cache = affine_lrelu_forward(out, self.params[f"W{i}"], self.params[f"b{i}"], {"alpha": self.alpha})
                cache_dict[f"layer{i}"] = cache

                if self.use_dropout:
                  out, cache = dropout_forward(out, self.dropout_param)
                  cache_dict[f"dropout{i}"] = cache


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        l2_reg = sum([np.sum(v**2) for k, v in self.params.items() if k.startswith("W")])
        loss, dout = softmax_loss(scores, y)
        loss = loss + (0.5 * self.reg * l2_reg)


        for i in reversed(range(1, self.num_layers+1)):
            if i == self.last_index:
                dout, grads[f"W{i}"], grads[f"b{i}"] = affine_backward(dout, cache_dict[f"layer{i}"])
            else:
                if self.use_dropout:
                  dout = dropout_backward(dout, cache_dict[f"dropout{i}"])
                dout, grads[f"W{i}"], grads[f"b{i}"] = affine_lrelu_backward(dout, cache_dict[f"layer{i}"])


        for param in self.params:
            if param.startswith("W"):
                grads[param] += self.reg * self.params[param]


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
