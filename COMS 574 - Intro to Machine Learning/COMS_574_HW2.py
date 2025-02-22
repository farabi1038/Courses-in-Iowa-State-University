import numpy as np
import matplotlib.pyplot as plt

import numpy  # import again
import matplotlib.pyplot  # import again

import numpy.linalg
import numpy.random

import hashlib


def generate_data(Para1, Para2, seed=0):
    """Generate binary random data

    Para1, Para2: dict, {str:float} for each class, 
      keys are mx (center on x axis), my (center on y axis), 
               ux (sigma on x axis), ux (sigma on y axis), 
               y (label for this class)
    seed: int, seed for NUMPy's random number generator. Not Python's random.

    """
    numpy.random.seed(seed)
    X1 = numpy.vstack((numpy.random.normal(Para1['mx'], Para1['ux'], Para1['N']),
                       numpy.random.normal(Para1['my'], Para1['uy'], Para1['N'])))
    X2 = numpy.vstack((numpy.random.normal(Para2['mx'], Para2['ux'], Para2['N']),
                       numpy.random.normal(Para2['my'], Para2['uy'], Para2['N'])))
    Y = numpy.hstack((Para1['y'] * numpy.ones(Para1['N']),
                      Para2['y'] * numpy.ones(Para2['N'])))
    X = numpy.hstack((X1, X2))
    X = numpy.transpose(X)
    return X, Y


def plot_data_hyperplane(X, y, w, filename):
    """
    X: 2-D numpy array, each row is a sample, not augmented 
    y: 1-D numpy array, the labels 
    w: 1-by-3 numpy array, the last element of which is the bias term

    Examples
    --------------

    >>> X = numpy.array([[1,2], \
                         [4,5], \
                         [7,8]]) 
    >>> y = numpy.array([1,-1,1])
    >>> w = [1, 2, -10]
    >>> filename = "test.png"
    >>> plot_data_hyperplane(X, y, w, filename)
    >>> hashlib.md5(open(filename, 'rb').read()).hexdigest()
    '242f39ee1cf8d7874609b98de8585fa0'
    """

    # your code here

    feature1 = X[y == 1]
    feature2 = X[y == -1]

    plt.plot(feature1[:, 0], feature1[:, 1], 'ro')
    plt.plot(feature2[:, 0], feature2[:, 1], 'bo')
    x_ticks = numpy.array([numpy.min(X[:, 0]), numpy.max(X[:, 0])])
    y_ticks = -1 * (x_ticks * w[0] + w[2]) / w[1]
    # y_ticks = -1 * (x_ticks * w[0]) / w[1]
    plt.plot(x_ticks, y_ticks)
    plt.xlim(numpy.min(X[:, 0]), numpy.max(X[:, 0]))
    plt.ylim(numpy.min(X[:, 1]), numpy.max(X[:, 1]))
    plt.savefig(filename)
    plt.show()
    matplotlib.pyplot.close('all')



def plot_data_hyperplane2(X, y, w, filename):
    """
    X: 2-D numpy array, each row is a sample, not augmented
    y: 1-D numpy array, the labels
    w: 1-by-3 numpy array, the last element of which is the bias term

    Examples
    --------------

    >>> X = numpy.array([[1,2], \
                         [4,5], \
                         [7,8]])
    >>> y = numpy.array([1,-1,1])
    >>> w = [1, 2, -10]
    >>> filename = "test.png"
    >>> plot_data_hyperplane(X, y, w, filename)
    >>> hashlib.md5(open(filename, 'rb').read()).hexdigest()
    '242f39ee1cf8d7874609b98de8585fa0'
    """

    # your code here

    feature1 = X[y == 1]
    feature2 = X[y == -1]

    plt.plot(feature1[:, 0], feature1[:, 1], 'ro')
    plt.plot(feature2[:, 0], feature2[:, 1], 'bo')
    x_ticks = numpy.array([numpy.min(X[:, 0]), numpy.max(X[:, 0])])
    #y_ticks = -1 * (x_ticks * w[0] + w[2]) / w[1]
    y_ticks = -1 * (x_ticks * w[0]) / w[1]
    plt.plot(x_ticks, y_ticks)
    plt.xlim(numpy.min(X[:, 0]), numpy.max(X[:, 0]))
    plt.ylim(numpy.min(X[:, 1]), numpy.max(X[:, 1]))
    plt.savefig(filename)
    plt.show()
    matplotlib.pyplot.close('all')




def plot_mse(X, y, filename):
    """
    X: 2-D numpy array, each row is a sample, not augmented
    y: 1-D numpy array

    Examples
    -----------------
    >>> X,y = generate_data(\
        {'mx':1,'my':2, 'ux':0.1, 'uy':1, 'y':1, 'N':20}, \
        {'mx':2,'my':4, 'ux':.1, 'uy':1, 'y':-1, 'N':50},\
        seed=10)
    >>> plot_mse(X, y, 'test1.png')
    array([-1.8650779 , -0.03934209,  2.91707992])
    >>> X,y = generate_data(\
    {'mx':1,'my':-2, 'ux':0.1, 'uy':1, 'y':1, 'N':20}, \
    {'mx':-1,'my':4, 'ux':.1, 'uy':1, 'y':-1, 'N':50},\
    seed=10)
    >>> # print (X, y)
    >>> plot_mse(X, y, 'test2.png')
    array([ 0.93061084, -0.01833983,  0.01127093])
    """
    w = np.array([0, 0, 0])  # just a placeholder

    # your code here
    ones = numpy.ones(X.shape[0]).reshape(X.shape[0], 1)

    X = numpy.concatenate((X, ones), axis=1)

    compound = numpy.matmul(X.T, X)
    inv = numpy.linalg.inv(compound)
    all_but_y = numpy.matmul(inv, numpy.transpose(X))

    w = numpy.matmul(all_but_y, numpy.transpose(y))

    # Plot after you have w.
    plot_data_hyperplane(X, y, w, filename)
    a, b, c = w[0], w[1], w[2]
    h = [0, -1 * c / a]
    vd = [-1 * c / b, 0]
    # print(h, vd)
    plt.plot(h, vd)
    plt.show()
    return w


def plot_fisher(X, y, filename):
    """
    X: 2-D numpy array, each row is a sample, not augmented
    y: 1-D numpy array

    Examples
    -----------------
    >>> X,y = generate_data(\
        {'mx':1,'my':2, 'ux':0.1, 'uy':1, 'y':1, 'N':20}, \
        {'mx':2,'my':4, 'ux':.1, 'uy':1, 'y':-1, 'N':50},\
        seed=10)
    >>> plot_fisher(X, y, 'test3.png')
    array([-1.61707972, -0.0341108 ,  2.54419773])
    >>> X,y = generate_data(\
        {'mx':-1.5,'my':2, 'ux':0.1, 'uy':2, 'y':1, 'N':200}, \
        {'mx':2,'my':-4, 'ux':.1, 'uy':1, 'y':-1, 'N':50},\
        seed=1)
    >>> plot_fisher(X, y, 'test4.png')
    array([-1.54593468,  0.00366625,  0.40890079])
    """




    # your code here
    f1 = X[y == 1]
    f2 = X[y == -1]

    mean1, mean2 = f1.mean(axis=0).reshape(-1, 1), f2.mean(axis=0).reshape(-1, 1)
    Sw = np.cov(f1.T) + np.cov(f2.T)
    inv_S = np.linalg.inv(Sw)
    w = inv_S.dot(mean1 - mean2)

    # Plot after you have w.
    plot_data_hyperplane2(X, y, w, filename)
    return w


if __name__ == "__main__":
    import doctest

    doctest.testmod()