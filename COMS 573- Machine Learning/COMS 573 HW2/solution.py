import sys
sys.path.append('libsvm/python/')
from svmutil import svm_train, svm_predict
'''
Homework4: support vector machine classifier

You need to use two functions 'svm_train' and 'svm_predict'
from libsvm library to start your homework. Please read the 
readme.txt file carefully to understand how to use these 
two functions.

'''

def svm_with_diff_c(train_label, train_data, test_label, test_data):
	'''
	Use 'svm_train' function with training label, data and different value 
	of cost c to train a svm classify model. Then apply the trained model
	on testing label and data.
	
	The value of cost c you need to try is listing as follow:
	c = [0.01, 0.1, 1, 2, 3, 5]
	Please keep other parameter options as default.
	No return value is needed
	'''

	c=['-c 0.01','-c 0.1','-c 1','-c 2','-c 3','-c 5']
	for i in range(0,len(c)):
		#libsvm_options = '-c 2'
		model = svm_train(train_label, train_data, c[i])
		predicted_label, test_acc, decision_values = svm_predict(test_label, test_data, model)
		print(test_acc)

def svm_with_diff_kernel(train_label, train_data, test_label, test_data):
	'''
	Use 'svm_train' function with training label, data and different kernel
	to train a svm classify model. Then apply the trained model
	on testing label and data.
	
	The kernel  you need to try is listing as follow:
	1. linear kernel
	2. polynomial kernel
	3. radial basis function kernel
	Please keep other parameter options as default.
	No return value is needed
	'''

	c = ['-t 1', '-t 2', '-t 3']
	print('starting kernels')
	for i in range(0, len(c)):
		# libsvm_options = '-c 2'
		model = svm_train(train_label, train_data, c[i])
		predicted_label, test_acc, decision_values = svm_predict(test_label, test_data, model)
		print(test_acc)
