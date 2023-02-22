import numpy as np
from random import shuffle
import builtins


def softmax_loss_naive(W, X, y, reg_l2, reg_l1 = 0):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg_l2: (float) regularization strength for L2 regularization
    - reg_l1: (float) default: 0. regularization strength for L1 regularization 
                to be used in Elastic Net Reg. if supplied, this function uses Elastic
                Net Regularization.

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    if reg_l1 == 0.:
        regtype = 'L2'
    else:
        regtype = 'ElasticNet'
    
    ##############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.      #
    # Store the loss in loss and the gradient in dW. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the         #
    # regularization! If regtype is set as 'L2' just implement L2 Regularization #
    # else implement both L2 and L1.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]

    def safelog(x):
        return np.log(x + 1e-100)
 

    for i in range(num_train):
        scores = X[i].dot(W)
        max_score = np.max(scores)
        safe_scores = scores - max_score
        exp_scores = np.exp(safe_scores) 

        denominator = np.sum(exp_scores)
        
        true_class_prob = exp_scores[y[i]] / denominator
        #print(f'shape true class prob: {true_class_prob.shape}')

        loss += - safelog(true_class_prob)
 
        d_loss_scores = - (exp_scores/denominator) * (true_class_prob)
        d_loss_scores[y[i]] = (true_class_prob) * (1 - (true_class_prob))
        #print(f'd_loss_scores shape is : {d_loss_scores[y[i]]}')

        d_loss_scores = -d_loss_scores * (1/(true_class_prob))
        
        #print(f'shape d_loss_scores is: {d_loss_scores.shape}')
        dW += np.outer(X[i], d_loss_scores)


    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    loss += (reg_l2 * np.sum(W * W)) + (reg_l1 * np.sum(np.abs(W)))
    dW += (2*reg_l2*W) + (reg_l1 * np.sign(W))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

def softmax_loss_vectorized(W, X, y, reg_l2, reg_l1 = 0):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    if reg_l1 == 0:
        regtype = 'L2'
    else:
        regtype = 'ElasticNet'
    
    ##############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.   #
    # Store the loss in loss and the gradient in dW. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the         #
    # regularization! If regtype is set as 'L2' just implement L2 Regularization #
    # else implement both L2 and L1.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def safelog(x):
        return np.log(x + 1e-100)

    #Implement vectorized softmax loss
    scores = np.matmul(X, W)
    safe_exp_scores = scores - np.max(scores, axis = 1).reshape(-1, 1)
    exp_scores = np.exp(safe_exp_scores)
    denominator = np.sum(exp_scores, axis=1)
    true_class_probs = (exp_scores[range(scores.shape[0]), y].reshape([scores.shape[0], 1])) / denominator.reshape([scores.shape[0], 1])
    log_probs = safelog(true_class_probs)
    loss = -np.sum(log_probs) / scores.shape[0]
    loss += (reg_l2 * np.sum(W * W)) + (reg_l1 * np.sum(np.abs(W)))

    #Implement vectorized softmax gradient
    log_gradient = -(1 / true_class_probs)
    softmax_gradient = - exp_scores / denominator.reshape(-1 , 1)
    softmax_gradient = softmax_gradient * true_class_probs
    softmax_gradient[range(scores.shape[0]), y] = np.squeeze((true_class_probs) * (1 - (true_class_probs)))
    softmax_gradient = softmax_gradient * log_gradient
    dW = np.matmul(X.T, softmax_gradient) / scores.shape[0]

    dW += (2*reg_l2*W) + (reg_l1 * np.sign(W))
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
