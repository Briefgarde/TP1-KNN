from collections import Counter

import numpy as np


def difference(A, B):
    """Computes the difference between two vectors.

    Inputs:
        - A: A vector of length same length as B
        - B: A vector of length same length as A

    Output:
        returns: the difference between A and B
    """
    assert len(A) == len(B), "A and B should have same shape"

    diff = A-B

    return diff


def inner_product(A, B):
    """Computes the inner product between two vectors.

    Inputs:
        - A: A vector of length same length as B
        - B: A vector of length same length as A

    Output:
        returns: the inner product
    """
    assert len(A) == len(B), "A and B should have same shape"
    inner_product = np.dot(A,B)

    return inner_product


def norm(A, B):
    """Computes the norm of two vectors.

    Inputs:
        - A: A vector of length same length as B
        - B: A vector of length same length as A

    Returns:
        - norm: the norm of vector AB
    """
    assert len(A) == len(B), "A and B should have same shape"

    
    norm = np.sqrt(np.dot(A,B))

    return norm


class KNearestNeighbors(object):
    """a kNN classifier with L2 distance."""

    def __init__(self):
        pass

    def train(self, X, y):
        """Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
        consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
            y[i] is the label for X[i].
        """
        ## insert your code here ##
        self.X_train = X 
        self.y_train = y
        #### end of your code ####

    def predict_label(self, query, k=1):
        """Given a query point, predicts the label according to the self.X_train dataset.

        Inputs:
        - query: the point to which the label should be predicted.
        - k: The number of nearest neighbors that vote for the predicted labels.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
        test data, where y[i] is the predicted label for the test point X[i].
        """
        label = None

        ## insert your code here ##
        # This method uses for loop : 
        #distances = np.array([norm(difference(query, instance), difference(query, instance)) for instance in self.X_train])
        # While it works just fine, performances are sloooooow
        # For reference : when calculating the decision surface with this method ^, it takes about 40-60 seconds.
        
        # Without loop method : 
        # With this method v, it's barely 5 seconds. 
        distances = np.sqrt(np.sum(np.square(self.X_train-query), axis=1))
        # How to calculate the distance without a for loop : 
        # matrice n,2 - vector 1,2 = doesn't work. WE need to make the shape of the vector - matrix the same. How ? 
        # We are going to stack the vector on itself, repeating the same vector n time. 
        # We don't need to actually stack the vector literally. Broadcasting in python/numpy let us write matrix-vector just fine
            # Broadcasting : we repeat the X_train-query operation at each row of the matrix
        # A mathematician would kill if you did that tho. The shape of the matrix and vector do not match, so we can't do it normally
        # To broadcast in this context, we do need to have at least one of the shape number correct
        # If it was matrix 4,5 and vector 1,2, it wouldn't work
        
        
        # This is the distance between the query and each point in the train dataset
        distances = np.reshape(distances, (distances.shape[0], 1)) 
        # This is the labels 
        labels_1d = np.reshape(self.y_train, (self.y_train.shape[0], 1))
        
        #I'm putting those two 1D array together, so I can order them and pick the k first neighbors.
        answersLabel = np.concatenate((labels_1d, distances), axis=1) 
        answersLabelOrdered = answersLabel[answersLabel[:,1].argsort()]
        labelsToVoteOn = answersLabelOrdered[:k, 0]
        label = np.unique(labelsToVoteOn, return_counts=True)[0][np.argmax(np.unique(labelsToVoteOn, return_counts=True)[1])]

        
        #### end of your code ####
        return label

    def predict(self, query_array, k=1):
        """Predict labels for test data using this classifier.

        Inputs:
        - query_array: A numpy array of shape (num_test, D) containing test data consisting
            of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.

        Returns:
        - labels: A array of shape (num_test,) containing predicted labels for the
        test data, where y[i] is the predicted label for the test point query_array[i].
        """
        labels = []

        ## insert your code here ##
        labels = [self.predict_label(query, k) for query in query_array]
        #### end of your code ####
        return labels
