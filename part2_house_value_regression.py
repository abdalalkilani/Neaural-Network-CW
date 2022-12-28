import torch
import random
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing, impute, metrics, model_selection
import torch.nn as nn
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import copy
from sklearn.base import BaseEstimator
import warnings
import random


class Regressor():

	@staticmethod
	def init_weights(CurrLayer):
		if isinstance(CurrLayer, nn.Linear):
			nn.init.xavier_uniform(CurrLayer.weight)
			CurrLayer.bias.data.fill_(0)

	def __init__(self, x, nb_epoch = 1000, neurons = [13, 12, 8], learning_rate=0.001, batch_size = 512, early_stop = 20):
		# You can add any input parameters you need
		# Remember to set them with a default value for LabTS tests
		""" 
		Initialise the model.
		Arguments:
			- x {pd.DataFrame} -- Raw input data of shape 
				(batch_size, input_size), used to compute the size 
				of the network.
			- nb_epoch {int} -- number of epochs to train the network.

		"""
		#######################################################################
		#					   ** START OF YOUR CODE **
		#######################################################################
		self.x = x
		self.xminmax = preprocessing.MinMaxScaler()
		self.yminmax = preprocessing.MinMaxScaler()
		
		# Replace this code with your own
		X, _ = self._preprocessor(x, training = True)
		self.input_size = X.shape[1]
		self.output_size = 1
		self.nb_epoch = nb_epoch
		self.neurons = neurons
		self.batch_size = batch_size
		# self.batch_size = X.shape[0]
		self.learning_rate = learning_rate
		self.mseloss = nn.MSELoss()

		self.early_stop = early_stop

		# make sure to change this up
		inpn = self.input_size
		layers=[]
		#where k is the individual layer
		# AHHHHHHHHHHHHHHHHHH
		for k in neurons:
			#try with ReLU or other activation function
			layers.append(nn.Linear(inpn, k))
			layers.append(nn.ReLU())
			# layers.append(nn.ReLU())
			inpn = k
		layers.append(nn.Linear(inpn, self.output_size))
		layers.append(nn.ReLU())

		self.model = nn.Sequential(*layers)
		self.model.apply(self.init_weights)
		self.model.double()

		return

		#######################################################################
		#					   ** END OF YOUR CODE **
		#######################################################################

	def _preprocessor(self, x, y = None, training = False):
		""" 
		Preprocess input of the network.
		  
		Arguments:
			- x {pd.DataFrame} -- Raw input array of shape 
				(batch_size, input_size).
			- y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
			- training {boolean} -- Boolean indicating if we are training or 
				testing the model.

		Returns:
			- {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
			  size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
			- {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
			  size (batch_size, 1).
			
		"""

		#######################################################################
		#					   ** START OF YOUR CODE **
		#######################################################################
		if training: 
			self.x = x

		impute_knn = KNNImputer(n_neighbors=2)
		# if training:
			#filling NaN values in text columns
		#x['ocean_proximity'] = x.loc[:, ['ocean_proximity']].fillna(value=x['ocean_proximity'].mode()[0])
		# else:
		#	 # this throws error, as cant fill NaN with None, needs a value (e.g value=0)
		#	 x['ocean_proximity'] = x.loc[:, ['ocean_proximity']].fillna(value=None)
		
		pd.options.mode.chained_assignment = None
		#one-hot encoding for textual values?
		OceanProx = preprocessing.LabelBinarizer().fit_transform(x['ocean_proximity'])
		x = x.drop('ocean_proximity', axis=1)
		x = x.join(pd.DataFrame(OceanProx))

		# if training:
		x = impute_knn.fit_transform(x)
		# print(pd.isna(x).sum())

		# min-max normalisation
		if training: 
			# x = (x-x.min())/(x.max()-x.min())
			self.xminmax.fit(x)
			if isinstance(y, pd.DataFrame):
				# y = (y-y.min())/(y.max()-y.min())
				self.yminmax.fit(y)

		x_df = pd.DataFrame(data=x)
		y_df = pd.DataFrame(data=y)
		x = torch.tensor(self.xminmax.transform(x_df.values))
		y = torch.tensor(self.yminmax.transform(y_df.values)) if isinstance(y, pd.DataFrame) else None

		# Replace this code with your own
		# Return preprocessed x and y, return None for y if it was None
		return x, y 

		#######################################################################
		#					   ** END OF YOUR CODE **
		#######################################################################

		
	def fit(self, x, y):
		"""
		Regressor training function

		Arguments:
			- x {pd.DataFrame} -- Raw input array of shape 
				(batch_size, input_size).
			- y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

		Returns:
			self {Regressor} -- Trained model.

		"""

		#######################################################################
		#					   ** START OF YOUR CODE **
		#######################################################################
		
		x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=31)

		X, Y = self._preprocessor(x_train, y_train, training = True) # Do not forget
		Xval, Yval = self._preprocessor(x_val, y_val, training = False)

		optim=torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
		# can also use Adam, AdaDelta - its worse
		X_size = X.size()[0]
		batch_size = self.batch_size
		if batch_size > X_size:
			batch_size = X_size
		current_best = self
		current_min = float("inf")
		current_best_index = 0
		for i in range(self.nb_epoch):

			# optim.zero_grad()
			# prediction = self.model(X)
			# loss = self.mseloss(prediction, Y)
			# loss.backward()
			# optim.step()



			for batch in range(0, X_size, batch_size):
				optim.zero_grad()

				X_batch = X[batch:batch+batch_size]
				y_batch = Y[batch:batch+batch_size]

				prediction = self.model(X_batch)
				loss = self.mseloss(prediction, y_batch)
				loss.backward()
				optim.step()

# VALIDATION LOSS - uncomment when want to do early stopping or print validation loss
			# print(x_val)
			# validation_loss = self.score(x_val, y_val)
			epoch_prediction = self.model(Xval)
			epoch_prediction = epoch_prediction.detach().numpy()
			y_pred = self.yminmax.inverse_transform(epoch_prediction)
			y_true = y_val.to_numpy()
			# print(y_true)
			validation_loss = metrics.mean_squared_error(y_true, y_pred, squared=False)
			print(validation_loss)
			if validation_loss < current_min:
				current_best_index = i
				current_best = copy.deepcopy(self)
				current_min = validation_loss
			else:
				if i - current_best_index > self.early_stop:
					self = current_best
					return self

		self = current_best
		return self



			# 	# print(len(prediction))
			# epoch_prediction = prediction.detach().numpy()
			# y_pred = self.yminmax.inverse_transform(epoch_prediction)
			# y_true = y.to_numpy()
			# # print(y_true)
			# score = metrics.mean_squared_error(y_true, y_pred, squared=False)
			# print(score)
# AHHHHH------------------
			# print(len(x_val))
			# print(len(y_val))
			# dev_loss = self.score(x_val, y_val)
			# index = 0
			# if dev_loss < current_min:
			# 	current_min = dev_loss
			# 	index = i
			# 	current_best = copy.deepcopy(self)
			# else:
			# 	if i - index > self.early_stop:
			# 		# this means if we dont improve our loss for 'early_stop' epochs, we just stop there and return last improvement
			# 		return current_best

		# X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
		# optim=torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

		# for i in range(self.nb_epoch):

		# 	optim.zero_grad()
		# 	prediction = self.model(X)
		# 	# loss = (prediction - Y).sum()
		# 	loss = self.mseloss(prediction, Y)
		# 	loss.backward()
		# 	optim.step()

		# return self

		#######################################################################
		#					   ** END OF YOUR CODE **
		#######################################################################

			
	def predict(self, x):
		"""
		Output the value corresponding to an input x.

		Arguments:
			x {pd.DataFrame} -- Raw input array of shape 
				(batch_size, input_size).

		Returns:
			{np.ndarray} -- Predicted value for the given input (batch_size, 1).

		"""

		#######################################################################
		#					   ** START OF YOUR CODE **
		#######################################################################

		X, _ = self._preprocessor(x, training = False) # Do not forget
		prediction = self.model(X).detach().numpy()
		y_pred = self.yminmax.inverse_transform(prediction)

		return y_pred

		#######################################################################
		#					   ** END OF YOUR CODE **
		#######################################################################

	def score(self, x, y):
		"""
		Function to evaluate the model accuracy on a validation dataset.

		Arguments:
			- x {pd.DataFrame} -- Raw input array of shape 
				(batch_size, input_size).
			- y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

		Returns:
			{float} -- Quantification of the efficiency of the model.

		"""

		#######################################################################
		#					   ** START OF YOUR CODE **
		#######################################################################
		
		X, Y = self._preprocessor(x, y, training = False) # Do not forget
		prediction = self.model(X).detach().numpy()
		
		y_pred = self.yminmax.inverse_transform(prediction)


		y_real = y.to_numpy()
		return metrics.mean_squared_error(y_real, y_pred, squared=False)

		#######################################################################
		#					   ** END OF YOUR CODE **
		#######################################################################

	def get_params(self, deep=True):
		return {
			'x': self.x,
			'learning_rate': self.learning_rate,
			'nb_epoch': self.nb_epoch,
			'neurons': self.neurons,
			'batch_size': self.batch_size
		}

	def set_params(self, **params):
		for param, value in params.items():
			setattr(self, param, value)
		
		return self


def save_regressor(trained_model): 
	""" 
	Utility function to save the trained regressor model in part2_model.pickle.
	"""
	# If you alter this, make sure it works in tandem with load_regressor
	with open('part2_model.pickle', 'wb') as target:
		pickle.dump(trained_model, target)
	print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
	""" 
	Utility function to load the trained regressor model in part2_model.pickle.
	"""
	# If you alter this, make sure it works in tandem with save_regressor
	with open('part2_model.pickle', 'rb') as target:
		trained_model = pickle.load(target)
	print("\nLoaded model in part2_model.pickle\n")
	return trained_model



def RegressorHyperParameterSearch(x_train, y_train, x_test, y_test): 
	# Ensure to add whatever inputs you deem necessary to this function
	"""
	Performs a hyper-parameter for fine-tuning the regressor implemented 
	in the Regressor class.

	Arguments:
		Add whatever inputs you need.
		
	Returns:
		The function should return your optimised hyper-parameters. 

	"""

	#######################################################################
	#					   ** START OF YOUR CODE **
	#######################################################################
	# x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
	# x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
	iterations = 5

	for it in range(iterations):
		max_error = np.inf
		batch_size = [128]
		epochs = [500, 1000, 2000]
		learning_rate = [0.001, 0.005]
		# hidden_layers = [1, 2, 3]
		max_neurons = [random.randint(3, 25) for _ in range(12)]
		neurons = max_neurons[:2]

		early_stop = [20]

		with open(f'result_{it}.out', 'w') as f:
			# param_grid = dict(
			# 	batch_size=batch_size, 
			# 	epochs=epochs, 
			# 	learning_rate=learning_rate, 
			# 	# hidden_layers=hidden_layers, 
			# 	neurons=neurons,
			# 	early_stop=early_stop)

			
			# Grid_Search = model_selection.GridSearchCV( 
			# 	Regressor(x=x),
			# 	param_grid = copy.deepcopy(param_grid),
			# 	scoring = 'neg_root_mean_squared_error',
			# 	return_train_score=True,
			# 	verbose = 3,
			# 	cv = 5,
			# 	n_jobs = -1,
			# 	refit = True
			# 	# pre_dispatch = ’2*n_jobs’
			# )
			# warnings.filterwarnings("ignore")
			# Grid_Search.fit(x, y)

			for b in batch_size:
				for e in epochs:
					for l in learning_rate:
						for ea in early_stop:
							kwargs = dict(
								batch_size=b, 
								nb_epoch=e, 
								learning_rate=l, 
								# hidden_layers=hidden_layers, 
								neurons=neurons,
								early_stop=ea)
							print(kwargs)

							regressor = Regressor(x_train, **kwargs)
							# regressor = Regressor(x, nb_epoch = 272, neurons = [13, 13, 12, 12, 10, 10], learning_rate=0.001, batch_size = 512, early_stop = 20)
							regressor.fit(x_train, y_train)
							# save_regressor(regressor)

							# Error
							error = regressor.score(x_test, y_test)
							print(f'error:{error}')

							f.write(f'---------------------------------------------------------------------------------------------------------------------------------\n')
							f.write(f'params: {kwargs}\n')
							f.write(f'Regressor error: {error if error != 226314.01986972374 else "dogshit"}\n')
							f.write(f'---------------------------------------------------------------------------------------------------------------------------------\n')
						
							if error < max_error:
								res = kwargs
								max_error = error


	return  res
	# Return the chosen hyper parameters

	#######################################################################
	#					   ** END OF YOUR CODE **
	#######################################################################

def build_and_test(i, x, y, kwargs):
	res1, res2 = [], []
	print('----------------------------------------------------------------------------------------------------------')
	with open(f'test2_{i}.out', 'w')as f:
		f.write(f'configuration: {kwargs}\n')
		for it in range(10):
			print(f'currently testing regressor {it+1} of 10')
			x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
			regressor = Regressor(x_train, **kwargs)
			regressor.fit(x_train, y_train)
			error = regressor.score(x_test, y_test)
			res1.append(error)
			if error != 226314.01986972374:
				res2.append(error)
			f.write(f'error:\t{error}\n')
		f.write(f'---------------------------------------------------------------------------------------------------------------------------------\n')
		f.write(f'configuration: {kwargs}\n')
		f.write(f'average error with outliers: {np.mean(res1)}\n')
		f.write(f'max error with outliers: {max(res1)}\n')
		f.write(f'min error with outliers: {min(res1)}\n')
		f.write(f'average error without outliers: {np.mean(res2)}\n')
		f.write(f'max error without outliers: {max(res2)}\n')
		f.write(f'min error without outliers: {min(res2)}\n')

def example_main():
	warnings.filterwarnings("ignore")

	# Use pandas to read CSV data as it contains various object types
	# Feel free to use another CSV reader tool
	# But remember that LabTS tests take Pandas DataFrame as inputs
	data = pd.read_csv("housing.csv") 
	output_label = "median_house_value"

	# Splitting input and output
	x = data.loc[:, data.columns != output_label]
	y = data.loc[:, [output_label]]
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


	# intitial_neurons = RegressorHyperParameterSearch(x_train, y_train, param_grid)

	# best_layer_ish = len(intitial_neurons['neurons'])
	# neurons = []
	# for _ in range(10):
	# 	neurons.append([random.randint(5,20) for _ in range(best_layer_ish)])

	# param_grid = dict(
	# 	batch_size=batch_size, 
	# 	epochs=epochs, 
	# 	learning_rate=learning_rate, 
	# 	# hidden_layers=hidden_layers, 
	# 	neurons=neurons,
	# 	early_stop=early_stop)

	# high_low = RegressorHyperParameterSearch(x_train, y_train, param_grid)

	# # (best_min, best_max) = min_max[0]
	# # for (index, (i,j)) in enumerate(min_max):
	# # 	low = i-min_max[0][0]
	# # 	high = j-min_max[0][1]
	# # 	if (low <= 0) and (high >= 0):
	# # 		best_min, best_max = min_max[index]

	# # Training
	# # This example trains on the whole available dataset. 
	# # You probably want to separate some held-out data 
	# # to make sure the model isn't overfitting
	# # regressor = RegressorHyperParameterSearch(x_train, y_train, param_grid=param_grid)


	# param_grid = dict(
	# 	batch_size=batch_size, 
	# 	epochs=epochs, 
	# 	learning_rate=learning_rate, 
	# 	# hidden_layers=hidden_layers, 
	# 	neurons=[high_low['neurons']],
	# 	early_stop=early_stop)

	# kwargs = RegressorHyperParameterSearch(x_train, y_train, x_test, y_test)
	# kwargs['nb_epoch'] = kwargs['epochs']
	# del kwargs['epochs']
	# print(kwargs)

	# regressor = Regressor(x_train, **kwargs)
	regressor = Regressor(x_train, nb_epoch = 5000, neurons = [30,20], learning_rate=0.005, batch_size = 100, early_stop = 500)
	regressor.fit(x_train, y_train)
	save_regressor(regressor)

	# Error
	error = regressor.score(x_test, y_test)
	print("\nRegressor error: {}\n".format(error))

	result = regressor.predict(x_test)
	print(result)

def final_main():
	warnings.filterwarnings("ignore")

	data = pd.read_csv("housing.csv") 
	output_label = "median_house_value"

	# Splitting input and output
	x = data.loc[:, data.columns != output_label]
	y = data.loc[:, [output_label]]

	configs = [
		{
			'batch_size': 128,
			'nb_epoch': 1000,
			'learning_rate': 0.005,
			'neurons': [25,9],
			'early_stop': 20
		},
		{
			'batch_size': 128,
			'nb_epoch': 2000,
			'learning_rate': 0.005,
			'neurons': [9,15],
			'early_stop': 20
		},
		{
			'batch_size': 128,
			'nb_epoch': 1000,
			'learning_rate': 0.005,
			'neurons': [23,4],
			'early_stop': 20
		},
		{
			'batch_size': 128,
			'nb_epoch': 2000,
			'learning_rate': 0.005,
			'neurons': [22,13],
			'early_stop': 20
		}
	]


	i = 0
	for con in configs:
		build_and_test(i, x, y, con)
		i+=1

if __name__ == "__main__":
	example_main()

