import math
import pandas as pd
import numpy as np
THETA = 0.5


class Tree(object):

    def __init__(self, feature=None, ys=[], left=None, right=None):
        self.feature = feature
        self.ys = ys
        self.left = left
        self.right = right

    @property
    def size(self):
        size = 1
        if type(self.left) == int:
            size += 1
        else:
            size += self.left.size
        if type(self.right) == int:
            size += 1
        else:
            size += self.right.size
        return size

    @property
    def depth(self):
        left_depth = 1 if type(self.left) == int else self.left.depth
        right_depth = 1 if type(self.right) == int else self.right.depth
        return max(left_depth, right_depth)+1


def entropy(data):
    """Compute entropy of data.

    Args:
        data: A list of data points [(x_0, y_0), ..., (x_n, y_n)]

    Returns:
        entropy of data (float)
    """
    ### YOUR CODE HERE
    classes, class_counts = np.unique(data, return_counts=True)
    entropy_value = np.sum([(-class_counts[i] / np.sum(class_counts)) * np.log2(class_counts[i] / np.sum(class_counts))
                            for i in range(len(classes))])
    return entropy_value


    ### END YOUR CODE


def gain(data, feature):
    """Compute the gain of data of splitting by feature.

    Args:
        data: A list of data points [(x_0, y_0), ..., (x_n, y_n)]
        feature: index of feature to split the data

    Returns:
        gain of splitting data by feature
    """
    ### YOUR CODE HERE
    total_entropy=entropy(d[1] for d in data)
    freqs={}
    for x,y in data:
        freqs[x[feature]]=freqs.get(x[feature],0)+1.0
    sub_entropy=0.0
    for key,value in freqs.items():
        sub_data=[y for x,y in data if x[feature]==key]
        sub_entropy+= value/len(data)*entropy(sub_data)
        return total_entropy-sub_entropy


    

    # please call self.entropy to compute entropy

    '''
    total_entropy = entropy(np.array(target))
    total = 0
    for v in data:
        total += sum(np.array(features)[v]) / sum(target) * entropy(np.array(features)[v])

    gain = total_entropy - total
    return gain
    # please call entropy to compute entropy

    '''
    ### END YOUR CODE


def get_best_feature(data):
    """Find the best feature to split data.

    Args:
        data: A list of data points [(x_0, y_0), ..., (x_n, y_n)]

    Returns:
        index of feature to split data
    """
    ### YOUR CODE HERE

    # please call gain to compute gain
    features = []
    target = []
    for i in data:
        features.append(i[0])
        target.append(i[1])
    # please call self.gain to compute gain
    item_values = [gain(data, feature) for feature in
                   range( len(data[0][0]))]  # Return the information gain values for the features in the dataset
    best_feature_index = np.argmax(item_values)




    return best_feature_index

    ### END YOUR CODE


def build_tree(data):
    ys = {}
    for x, y in data:
        ys[y] = ys.get(y, 0) + 1
    if len(ys) == 1:
        return list(ys)[0]
    feature = get_best_feature(data)
    left = [d for d in data if d[0][feature] <= THETA]
    right = [d for d in data if d[0][feature] > THETA]


    temp=[]
    target=[]

    for i in data:
        temp.append(i[0])
        target.append(i[1])
    temp=pd.DataFrame(temp)
    target=np.unique(target,return_counts=True)
    place= np.unique(temp[feature])

    par=int(place[np.argmax(target[1])])

    if len(place)<=1:
        return par
    elif len(data)==0 or temp.shape[1]==0:
        return par

    if len(left) == 0 or len(right) == 0:
        return par

    left_tree=build_tree(left)
    right_tree=build_tree(right)

    # Use THETA to split the continous feature


    ### END YOUR CODE
    return Tree(feature, ys, left_tree, right_tree)


def test_entry(tree, entry):
    x, y = entry
    if type(tree) == int:
        return tree, y
    if x[tree.feature] < THETA:
        return test_entry(tree.left, entry)
    else:
        return test_entry(tree.right, entry)


def test_data(tree, data):
    count = 0
    for d in data:
        y_hat, y = test_entry(tree, d)
        count += (y_hat == y)
    return round(count/float(len(data)), 4)


def prune_tree(tree, data):
    """Find the best feature to split data.

    Args:
        tree: a decision tree to prune
        data: A list of data points [(x_0, y_0), ..., (x_n, y_n)]

    Returns:
        a pruned tree
    """
    ### YOUR CODE HERE
    if type(tree) == int or len(data) == 0:
        return tree
    # feature = get_best_feature(data)
    feature = tree.feature
    major_class = 0 if tree.ys[0] > tree.ys[1] else 1
    left = [d for d in data if d[0][feature] <= THETA ]
    right = [d for d in data if d[0][feature] > THETA]
    prevdious_error = test_data(tree, data)
    # target = []
    #
    # for i in data:
    #     target.append(i[1])
    # major_class= np.bincount(target).argmax()
    after_error = test_data(major_class, data)
    if (after_error > prevdious_error):
        return major_class
    else:
        tree.left = prune_tree(tree.left, left)
        tree.right = prune_tree(tree.right, right)
    return tree

    # please call test_data to obtain validation error
    # please call prune_tree recursively for pruning tree

    ### END YOUR CODE
