import numpy as np
from random import shuffle

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
  num_label = W.shape[1]
  scores = np.zeros((num_train, num_label))
  for i in xrange(num_train):
      scores[i] = X[i].dot(W)
      scores[i] = scores[i] - np.max(scores[i])
      correct_score = scores[i,y[i]]
      loss += -np.log( np.exp(correct_score) / np.sum(np.exp(scores[i])) )
      for j in xrange(num_label):
          if j == y[i]:
              dW[:,j] += (np.exp(scores[i,j]) / np.sum(np.exp(scores[i])) - 1) * X[i]
          else:
              dW[:,j] += (np.exp(scores[i,j]) / np.sum(np.exp(scores[i])) ) * X[i]
      
  loss /= num_train
  dW /= num_train
  
  reg_loss = 0.5 * reg * np.sum(W**2)
  loss += reg_loss
  dW += reg * W
  pass
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
  '''
  num_train = X.shape[0]
  num_label = W.shape[1]
  scores = X.dot(W)
  max_scores = np.max(scores, axis = 1)[:, np.newaxis]
  scores = scores - max_scores
  correct_score = scores[range(num_train), y]
  pro =  np.exp(correct_score) / np.sum(np.exp(scores), axis=1) 
  loss = np.sum(-np.log(pro))
  loss /= num_train
  #print (np.exp(scores[0]) / np.sum(np.exp(scores[0])))[:, np.newaxis]
  for i in xrange(num_train):
    #dW += pro[i] * X[i][:,np.newaxis]
    #print X[i]*(np.exp(scores[i]) / np.sum(np.exp(scores[i]),axis=1)[np.newaxis,:]).shape
    dW += X[i][:, np.newaxis]*(np.exp(scores[i]) / np.sum(np.exp(scores[i])))[ np.newaxis,:]
    dW[:, y[i]] -= X[i]
  
              
  reg_loss = 0.5 * reg * np.sum(W**2)
  loss += reg_loss
  dW /= num_train
  dW += reg * W
  pass
  '''      
  num_train, dim = X.shape

  scores = X.dot(W)    # N by C
  # Considering the Numeric Stability
  scores_max = np.reshape(np.max(scores, axis=1), (num_train, 1))   # N by 1
  prob = np.exp(scores - scores_max) / np.sum(np.exp(scores - scores_max), axis=1, keepdims=True)
  y_trueClass = np.zeros_like(prob)
  y_trueClass[range(num_train), y] = 1.0    # N by C
  loss += -np.sum(y_trueClass * np.log(prob)) / num_train + 0.5 * reg * np.sum(W * W)
  print loss
  dW = np.dot(X.T, prob - y_trueClass) / num_train + reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

