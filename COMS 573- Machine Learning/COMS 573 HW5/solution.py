import argparse
import torch.nn as nn
import torch.nn.functional as func
def get_args():
    parser = argparse.ArgumentParser(description='Args for training networks')
    parser.add_argument('-seed', type=int, default=1, help='random seed')
    parser.add_argument('-num_epochs', type=int, default=20, help='num epochs')
    parser.add_argument('-batch', type=int, default=16, help='batch size')
    parser.add_argument('-lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('-drop', type=float, default=0.3, help='drop rate')
    args, _ = parser.parse_known_args()
    return args


class LeNet(nn.Module):
    def __init__(self, args):
        super(LeNet, self).__init__()
        ### YOUR CODE HERE

        self.convulation1 = nn.Conv2d(3, 6, kernel_size=5)
        self.convulation1_bn = nn.BatchNorm2d(6)
        self.convulation2 = nn.Conv2d(6, 16, kernel_size=5)
        self.convulation2_bn = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.drop = nn.Dropout(0.3)

        ### END YOUR CODE

    def forward(self, x):
        '''
        Input x: a batch of images (batch size x 3 x 32 x 32)
        Return the predictions of each image (batch size x 10)
        '''
        ### YOUR CODE HERE
        x = func.relu(self.convulation1(x))
        x = func.max_pool2d(x, 2)
        x = self.convulation1_bn(x)
        x = func.relu(self.convulation2(x))
        x = func.max_pool2d(x, 2)
        x = self.convulation2_bn(x)
        x = x.view(x.size(0), -1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x



        ### END YOUR CODE
