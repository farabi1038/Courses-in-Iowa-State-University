import numpy as np
import sys
from helper import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron as p
def show_images(data):
    """Show the input images and save them.

    Args:
        data: A stack of two images from train data with shape (2, 16, 16).
              Each of the image has the shape (16, 16)

    Returns:
        Do not return any arguments. Save the plots to 'image_1.*' and 'image_2.*' and
        include them in your report
    """
    ### YOUR CODE HERE

    for d in data:
        print(d.shape)
        plt.plot(d[0], d[1], 'b--')
        plt.show()

    ### END YOUR CODE


def show_features(X, y, save=True):
    """Plot a 2-D scatter plot in the feature space and save it. 

    Args:
        X: An array of shape [n_samples, n_features].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        save: Boolean. The function will save the figure only if save is True.

    Returns:
        Do not return any arguments. Save the plot to 'train_features.*' and include it
        in your report.
    """
    ### YOUR CODE HERE
    fig, ax = plt.subplots()
    X1 = X[:, 0]
    X2 = X[:, 1]
    ax.scatter(X1[y == 1], X2[y == 1], marker='*', c="red")
    ax.scatter(X1[y == -1], X2[y == -1], marker='+', c="blue")
    plt.show()
    if (save):
       plt.savefig(r'train_features.png')

    ### END YOUR CODE


class Perceptron(object):
    
    def __init__(self, max_iter):
        self.max_iter = max_iter

    def fit(self, X, y):
        """Train perceptron model on data (X,y).

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        ### YOUR CODE HERE

        self.W=np.ones(X.shape[1])
        w = np.ones(X.shape[1])
        accuracy =0
        max_accuracy = 0
        # for all epochs
        for i in range(self.max_iter):
            for i, j in zip(X, y):

                y_pred = 1 if (1 / (1 + np.exp(-(np.dot(w, i)))) >= 0.5) else -1
                #print("predicted value: ",y_pred,"actual value: ",int(j))
                #print("value of prediction",type(y_pred))
                if int(j) == 1 and y_pred == -1:
                    w = w + j *i
                elif int(j) == -1 and y_pred == 1:
                    w = w - j * i




            accuracy = self.score(self.predict(X), y)

            if (accuracy > max_accuracy):
                max_accuracy = accuracy
                self.W = w

        # After implementation, assign your weights w to self as below:
        self.W = w
        
        ### END YOUR CODE
        
        return self

    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W

    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
        ### YOUR CODE HERE
        Y = []
        for i in X:
            #print("predict test",(np.dot(self.W, i)))

            result = 1 if (1 / (1 + np.exp(-1*(np.dot(self.W, i).all()))) >= 0.5) else -1

            Y.append(result)
        return np.array(Y)

        ### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
        ### YOUR CODE HERE

        pred=self.predict(X)
        count=0
        for i in range(len(y)):
            if pred[i]==y[i]:
                count=count+1
        score=count/len(y)
        #print('score is',score)
        return score
        ### END YOUR CODE




def show_result(X, y, W):
    """Plot the linear model after training. 
       You can call show_features with 'save' being False for convenience.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        W: An array of shape [n_features,].
    
    Returns:
        Do not return any arguments. Save the plot to 'result.*' and include it
        in your report.
    """
    ### YOUR CODE HERE
    print(X.shape)
    min = np.min(X[:, 0])
    max = np.max(X[:, 0])
    x1 = np.linspace(min, max, 100)

    def x2(x1, w):
        w0 = w[0]
        w1 = w[1]
        x2 = []
        for i in range(0, len(x1 - 1)):
            x2_temp = (- w0 * x1[i]) / w1
            x2.append(x2_temp)
        return x2

    x_2 = np.asarray(x2(x1, W))

    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.plot(x1, x_2)
    plt.show()

    ### END YOUR CODE



def test_perceptron(max_iter, X_train, y_train, X_test, y_test):

    # train perceptron
    model = Perceptron(max_iter)
    print("data type",type(X_train),type(y_train))
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    W = model.get_params()
    print("weight of W",W)
    # test perceptron model
    test_acc = model.score(X_test, y_test)

    return W, train_acc, test_acc