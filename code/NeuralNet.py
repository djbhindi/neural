import numpy as np
import random
import math

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def sigmoid_deriv(x):
	g = 1 / (1 + math.exp(-x))
	return g*(1-g)

def tanh_deriv(x):
	g = math.tanh(x)
	return 1-(math.pow(g, 2))

class NeuralNet:
	def __init__():
		self.w1 = []
		self.w2 = []
		self.si = []

		self.activation_lower = math.tanh
		self.activation_upper = sigmoid

		self.deriv_lower = tanh_deriv
		self.deriv_upper = sigmoid_deriv


	def compute_stochastic_update(self, x_0, label):
		# 1 x 200
		s_1 = np.dot(x_0, self.w1) 
		x_1 = np.array([np.apply_along_axis(self.activation_lower, 0, s_1)])

		# 1 x 10
		s_2 = np.dot(x_1, self.w2)
		x_2 = np.array([np.apply_along_axis(self.activation_upper, 0, s_2)])

		s2_derivs = self.compute_upper_derivatives(s_2, x_2, vectorize_label(label))
		w2_update = np.dot(x_1.T, s2_derivs)

		s1_derivs = self.compute_lower_derivatives(s_1, x_1, s2_derivs, self.w2)
		w1_update = np.dot(x_0.T, s1_derivs)

		self.w1 = self.apply_update(self.w1, w1_update)
		self.w2 = self.apply_update(self.w2, w2_update)

	def compute_upper_derivatives(self, s_2, x_2, y):
		activation_derivs = np.array([np.apply_along_axis(self.deriv_upper, 0, s_2)])
		loss_vector = np.subtract(x_2, y)

		return np.multiply(activation_derivs, loss_vector)

	def vectorize_label(self, label):
		vec = np.zeros(9)
		vec[label] = 1
		return np.array([vec])

	def compute_lower_derivatives(self, s_1, x_1, s2_derivs, w2):
		activation_derivs = np.array([np.apply_along_axis(self.deriv_lower, 0, s_1)])
		back_propogation_terms = np.dot(s2_derivs, w2.T)

		return np.multiply(activation_derivs, back_propogation_terms)

	def apply_update(self, weights, update, stepsize):
		return np.subtract(weights, np.multiply(stepsize, update))

	def predict_point(self, point):
		s_1 = np.dot(point, self.w1) 
		x_1 = np.apply_along_axis(self.activation_lower, 0, s_1)

		s_2 = np.dot(x_1, self.w2)
		x_2 = np.apply_along_axis(self.activation_upper, 0, s_2)

		return np.argmax(x_2)