
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import pickle
import gzip


def structure_data(path):
	data = pd.read_csv(path)
	data = data.drop(columns='timestamp')
	data = data.to_numpy()    
	np.random.shuffle(data)
	train_data = np.split(data, [0,(data.shape[0]*80//100)])[1]
	test_data = np.split(data, [0,(data.shape[0]*80//100)])[2]
	return data, train_data, test_data

data, train_data, test_data = structure_data('ratings.csv')

def get_indices():
	user_to_index = {}
	index_to_user = []
	movie_to_index = {}
	index_to_movie = []

	for user_id, movie_id, rating in data:
		if user_id not in user_to_index:
			index_user = len(index_to_user)
			index_to_user.append(user_id)
			user_to_index[user_id] = index_user
		else:
			index_user = user_to_index[user_id]

		if movie_id not in movie_to_index:
			index_movie = len(index_to_movie)
			index_to_movie.append(movie_id)
			movie_to_index[movie_id] = index_movie
		else:
			index_movie = movie_to_index[movie_id]
	  

	return user_to_index, index_to_user, movie_to_index, index_to_movie


user_to_index, index_to_user, movie_to_index, index_to_movie = get_indices()


def get_sparse(data):
  data_by_user = []
  data_by_movie = []
  for i in range(len(user_to_index)):
	data_by_user.append([])

  for i in range(len(movie_to_index)):
	data_by_movie.append([])

  for user_id, movie_id, rating in data:
	index_user = user_to_index[user_id]
	index_movie = movie_to_index[movie_id]
	data_by_user[index_user].append((index_movie, rating))
	data_by_movie[index_movie].append((index_user, rating))
  return data_by_user, data_by_movie



user_rating, movie_rating = get_sparse(data)


user_train, movie_train =  get_sparse(train_data)
user_test, movie_test =  get_sparse(test_data)


data_dict = {"user_train": user_train, "movie_train": movie_train, "user_test": user_test}


with gzip.open('datas.pkl', 'wb') as f:
	pickle.dump(data_dict, f)


with gzip.open('datas.pkl', 'rb') as f:
	test = pickle.load(f)


user_train = test['user_train']
user_test  = test['user_test']
movie_train  = test['movie_train']


from tqdm import tqdm as tq
from concurrent.futures import ThreadPoolExecutor, wait



class Model:
	""" ALS Model implementation"""
	def __init__(self, data, user_test, lamda=0.001, thau=0.001, gamma=0.001, latent_dim=1):
		user_train, movie_train = data
		self.user_train = user_train
		self.user_test = user_test
		self.movie_train = movie_train
		self.lamda = lamda
		self.gamma = gamma
		self.thau = thau
		self.latent_dim = latent_dim
		self.num_user = len(user_train)
		self.num_item = len(movie_train)
		

		self.user_biases = np.zeros(self.num_user)
		self.item_biases = np.zeros(self.num_item)
		self.U = np.random.normal(0, 1./np.sqrt(latent_dim), size=(self.num_user, latent_dim))
		self.V = np.random.normal(0, 1./np.sqrt(latent_dim), size=(self.num_item, latent_dim))
		
	   
	def update_user(self, range):
		"""Update user_biases and user_matrix keeping item_biases and item_matrix fixed"""
		for m in range:
			bias = 0
			item_counter = 0
			for n, r in self.user_train[m]:
				bias += self.lamda*(float(r) - self.item_biases[n])
				item_counter +=1

			bias = bias / (self.lamda * item_counter + self.gamma)
			self.user_biases[m] = bias

			outer_sum = 0
			err = np.zeros(self.latent_dim)
			for n, r in self.user_train[m]:
				outer_sum += np.outer(self.V[n], self.V[n].T)
				err += (float(r) - self.user_biases[m] - self.item_biases[n])*self.V[n]

			self.U[m] = np.linalg.inv(self.lamda*outer_sum + self.thau*np.identity(self.latent_dim))@(self.lamda*err)
		
	def update_item(self, range):
		"""Update item_biases and item_matrix keeping user_biases and user_matrix fixed"""
		for n in range:
			bias = 0
			user_counter = 0
			for m, r in self.movie_train[n]:
				bias += self.lamda*(float(r) - self.user_biases[m])
				user_counter +=1

			bias = bias / (self.lamda * user_counter + self.gamma)
			self.item_biases[n] = bias
			#update for item matrix
			outer_sum = 0
			err = np.zeros(self.latent_dim)

			for m, r in self.movie_train[n]:
				outer_sum +=np.outer(self.U[m], self.U[m].T)
				err += (float(r) - self.user_biases[m] - self.item_biases[n])*self.U[m]

			self.V[n] = np.linalg.inv(self.lamda*outer_sum + self.thau*np.identity(self.latent_dim))@(self.lamda*err)
			
	def train(self, epoch):

		r1 = [i for i in range(self.num_user)]
		r2 = [i for i in range(self.num_item)]

		for _ in tq(range(epoch)):
			
			with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
				job1 = executor.submit(self.update_user, r1)
				#wait([job1])
				job2 = executor.submit(self.update_item, r2)            
			
		return self.U, self.V, self.user_biases, self.item_biases  
 

model = Model(data=(user_train, movie_train), user_test=user_test, latent_dim=2)


U, V, user_bias, item_bias = model.train(10)

movies = pd.read_csv('movies.csv')

# This gives us the list of movies with title 'The Lord of The Rings'

movies[movies['title'].str.contains('lord of the rings', case=False)]

# One of 'The Lord of The Rings' movie has movieId 2116
movie_to_index[116]
# 1903
# Let u be a dummy variable 
u = V[1903]

predicted_ratings = np.dot(u, V.T) + item_bias
recommendations = np.argsort(predicted_ratings)[::-1]
top3_recommendations = recommendations[:3]

top3_recommendations
# Returns array([19410, 53894, 26418]) -> What is this? movieId or index? Guess it is index

movie_top3 = [index_to_movie[id] for id in top3_recommendations] # This gives list of movieIDs

[movies[movies['movieId']==id]['title'] for id in index_top3]
# Returns [61209    Inside Out (1991)
 ''' Name: title, dtype: object,
 	# 34720    Ittefaq
 	Name: title, dtype: object,
 	36113    All About My Wife (2012)
 	Name: title, dtype: object] '''
# No 'Lord of The rings' in the recommended items

# Can adding the feature vectors improves the model? or does reducing the ite bias help?
predicted_ratings_two = np.dot(u, V.T) + 0.05*item_bias

