import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in xrange(num_train):
    scores = np.dot( X[i],W )
    shift_scores = scores - max(scores)
    tmp_sum = np.sum( np.exp(shift_scores) )
    loss_i = np.log( tmp_sum ) - shift_scores[y[i]]

    loss += loss_i
    for j in xrange(num_class):
      dW[:,j] += (np.exp(shift_scores[j])/tmp_sum )*X[i]
      if j == y[i]:
        dW[:,j] -= X[i]

  loss /= num_train
  loss += 0.5*reg*np.sum( W*W )
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = np.dot( X,W )
  shift_scores = scores - np.max( scores,axis=1 ).reshape(-1,1)
  softmax = np.exp(shift_scores) / np.sum( np.exp(shift_scores),axis=1 ).reshape(-1,1)
  loss -= np.sum( np.log( softmax[ range(num_train),list(y) ] ) )
  loss /= num_train

  softmax[range(num_train),list(y)] += -1
  dW = np.dot( X.T,softmax )/num_train

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

