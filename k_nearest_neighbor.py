import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y

    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      for j in xrange(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        dists[i,j] = np.sqrt( np.sum( np.square( X[i] - self.X_train[j] ) ) )
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists


  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      dists[i,:] = np.sqrt( np.sum( np.square( X[i] - self.X_train[:] ),axis = 1 ) )
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    dists = np.multiply(np.dot(X, self.X_train.T), -2)
    sq1 = np.sum(np.square(X), axis = 1,keepdims = True)
    sq2 = np.sum(np.square(self.X_train), axis=1)
    dists = np.add(dists, sq1)
    dists = np.add(dists, sq2.T)
    dists = np.sqrt(dists)
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      closest_y = self.y_train[np.argsort( dists[i,:] )[:k]]  #np.argsort:return the indices
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      y_pred[i] = np.argmax( np.bincount(closest_y) )
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred


def main():
  num_folds = 5
  k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

  X_train_folds = []
  y_train_folds = []
  ################################################################################
  # TODO:                                                                        #
  # Split up the training data into folds. After splitting, X_train_folds and    #
  # y_train_folds should each be lists of length num_folds, where                #
  # y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
  # Hint: Look up the numpy array_split function.                                #
  ################################################################################
  # pass
  y_train_ = y_train.reshape(-1, 1)
  X_train_folds, y_train_folds = np.array_split(X_train, 5), np.array_split(y_train_, 5)
  ################################################################################
  #                                 END OF YOUR CODE                             #
  ################################################################################

  # A dictionary holding the accuracies for different values of k that we find
  # when running cross-validation. After running cross-validation,
  # k_to_accuracies[k] should be a list of length num_folds giving the different
  # accuracy values that we found when using that value of k.
  k_to_accuracies = {}

  ################################################################################
  # TODO:                                                                        #
  # Perform k-fold cross validation to find the best value of k. For each        #
  # possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
  # where in each case you use all but one of the folds as training data and the #
  # last fold as a validation set. Store the accuracies for all fold and all     #
  # values of k in the k_to_accuracies dictionary.                               #
  ################################################################################
  # pass
  for k_ in k_choices:
    k_to_accuracies.setdefault(k_, [])
  for i in range(num_folds):
    classifier = KNearestNeighbor()
    X_val_train = np.vstack(X_train_folds[0:i] + X_train_folds[i + 1:])
    y_val_train = np.vstack(y_train_folds[0:i] + y_train_folds[i + 1:])
    y_val_train = y_val_train[:, 0]
    classifier.train(X_val_train, y_val_train)
    for k_ in k_choices:
      y_val_pred = classifier.predict(X_train_folds[i], k=k_)
      num_correct = np.sum(y_val_pred == y_train_folds[i][:, 0])
      accuracy = float(num_correct) / len(y_val_pred)
      k_to_accuracies[k_] = k_to_accuracies[k_] + [accuracy]
  ################################################################################
  #                                 END OF YOUR CODE                             #
  ################################################################################

  # Print out the computed accuracies
  for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
      print
      'k = %d, accuracy = %f' % (k, accuracy)
  # plot the raw observations
  for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

  # plot the trend line with error bars that correspond to standard deviation
  accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
  accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accuracies.items())])
  plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
  plt.title('Cross-validation on k')
  plt.xlabel('k')
  plt.ylabel('Cross-validation accuracy')
  plt.show()
  # Based on the cross-validation results above, choose the best value for k,
  # retrain the classifier using all the training data, and test it on the test
  # data. You should be able to get above 28% accuracy on the test data.
  best_k = 10

  classifier = KNearestNeighbor()
  classifier.train(X_train, y_train)
  y_test_pred = classifier.predict(X_test, k=best_k)

  # Compute and display the accuracy
  num_correct = np.sum(y_test_pred == y_test)
  accuracy = float(num_correct) / num_test
  print
  'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)