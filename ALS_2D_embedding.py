# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import pickle
import gzip


movies = pd.read_csv('movies.csv')
movies = movies.to_numpy()


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
	features = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", 
				"Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", 
				"War", "Western", "IMAX", "(no genres listed)"]
	feature_to_index = {} 
	user_to_index = {}
	index_to_user = []
	movie_to_index = {}
	index_to_movie = []

	for i in range(len(features)):
		feature_to_index[features[i]] = i


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
		

	return user_to_index, index_to_user, movie_to_index, index_to_movie, feature_to_index


user_to_index, index_to_user, movie_to_index, index_to_movie, feature_to_index = get_indices()


user_train, movie_train =  get_sparse(data)
user_test, movie_test =  get_sparse(test_data)


def get_sparse_features(data):
	data_by_feature = []
	data_movie_feature = []

	for i in range(len(movie_to_index)):    
		data_movie_feature.append([])
	
	for i in range(len(feature_to_index)):
		data_by_feature.append([])
	
	for movie_id, title, genres in data:
		features_list = genres.split("|")
		
		for feature in features_list:
			feature_index = feature_to_index[feature]
			data_by_feature[feature_index].append(movie_id)
			if movie_id not in movie_to_index:
				continue
			else:
				movie_index = movie_to_index[movie_id]
			data_movie_feature[movie_index].append(feature_index)

	return data_by_feature, data_movie_feature


data_by_feature, data_movie_feature = get_sparse_features(movies)

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
	def __init__(self, data, user_test, lamda=0.1, thau=0.1, gamma=0.1, latent_dim=5):
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
		#self.F = np.random.normal(0, 1./np.sqrt(latent_dim), size=(self.num_feature, latent_dim))


		self.cost_train = []
		self.cost_test = []
		self.rmse_train= []
		self.rmse_test = []

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
			
			loss, rms = self.get_loss_rmse(self.user_train)
			loss_test, rms_test = self.get_loss_rmse(self.user_test)
			self.cost_train.append(loss)
			self.cost_test.append(loss_test)
			self.rmse_train.append(rms)
			self.rmse_test.append(rms_test)
		return self.V
		
		
	def plot_cost(self):
		fig, axs = plt.subplots(figsize=(5, 4))  
		x = [i for i in range(len(self.cost_train))]
		y = [-i for i in self.cost_train]
		y2 = [-i for i in self.cost_test]
		
		axs.plot(x, y, x, y2)
		axs.set_xlabel('# of iterations, \n K = {}, lambda = {}, tau = {}, gamma  = {}'.format(self.latent_dim, self.lamda, self.thau, self.gamma ))
		axs.set_ylabel('Cost function')  
		axs.legend(["Cost train", "Cost test"])         
		axs.grid(True, linestyle='--', linewidth=0.5)
		
	def plot_RMSE(self):
		fig, axs = plt.subplots(figsize=(5, 4))
		x = [i for i in range(len(self.rmse_train))]        
		y = [i for i in self.rmse_train]
		y2 = [i for i in self.rmse_test]  

		axs.plot(x, y, x, y2)
		axs.set_xlabel('# of iterations, \n K = {}, lambda = {}, tau = {}, gamma  = {}'.format(self.latent_dim, self.lamda, self.thau, self.gamma))
		axs.set_ylabel('RMSE')       
		axs.legend(["RMSE train", "RMSE test"])         
		axs.grid(True, linestyle='--', linewidth=0.5)
		
		
	def get_loss_rmse(self, data):

		reg2 = 0
		for m in range(self.num_user):
			reg2 += (-self.thau/2)*(np.dot(self.U[m], self.U[m].T))

		reg3 = 0
		for n in range(self.num_item):
			reg3 += (-self.thau/2)*(np.dot(self.V[n], self.V[n].T))

		loss = 0
		sum_ = 0
		reg1 = 0
		RMSE_ = 0
		RMSE_counter = 0

		for m in range(self.num_user):
			for n, r in data[m]:
				err_sq = (float(r) - (np.dot(self.U[m], self.V[n]) + self.user_biases[m] + self.item_biases[n]))**2                
				sum_ += (-self.lamda/2)*err_sq
				RMSE_ += err_sq
				RMSE_counter +=1
		RMSE_total = np.sqrt(RMSE_/RMSE_counter)

		reg1 = (-self.gamma/2)*(np.dot(self.user_biases, self.user_biases.T) + np.dot(self.item_biases, self.item_biases.T))

		loss = sum_ + reg1 + reg2 + reg3
		return loss, RMSE_total    



model = Model(data=(user_train, movie_train), user_test=user_test, latent_dim=2)

V = model.train(10)

# 2D embedding with text label

def plot_embedding():
	features = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", 
				"Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", 
				"War", "Western", "IMAX", "(no genres listed)"]

	movie_id = [data_by_feature[i][j] for i in range(19) for j in range(5) ]
	movie_index = [movie_to_index[id] for id in movie_id]
	x = [V[i][0] for i in movie_index]
	y = [V[i][1] for i in movie_index]        
	z = [features[i] for i in range(19)]
	genre_list = []
	for el in z:
		for i in range(5):
			genre_list.append(el)
		
	c = []
	for i in range(19):
		for j in range(5):
			c.append(i*10)
	colors = np.array(c)
	plt.figure(figsize=(8, 6))
	plt.scatter(x, y, c=colors, cmap='viridis')

	# Add text labels to each point
	for i, label in enumerate(genre_list):
		plt.text(x[i], y[i], label, fontsize=10, ha='center', va='bottom')

	plt.title('2D Embeddings Scatter Plot with Text Labels')
	plt.xlabel('X-axis')
	plt.ylabel('Y-axis')
	
	# Show the plot
	plt.show()


# 2D embedding with color
def plot_embedding2():
	features = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", 
				"Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", 
				"War", "Western", "IMAX", "(no genres listed)"]

	movie_id = [data_by_feature[i][j] for i in range(19) for j in range(15) ]
	movie_index = [movie_to_index[id] for id in movie_id]
	x = [V[i][0] for i in movie_index]
	y = [V[i][1] for i in movie_index]        
	z = [features[i] for i in range(19)]
	genre_list = []
	for el in z:
		for i in range(15):
			genre_list.append(el)
		
	c = []
	for i in range(19):
		for j in range(15):
			c.append(i*10)
	colors = np.array(c)
	plt.figure(figsize=(8, 6))
	scatter = plt.scatter(x, y, c=colors, cmap='viridis', label='Genres')
	cbar = plt.colorbar(scatter, orientation='vertical')

	cbar.set_ticks([i*10 for i in range(19)])  # Adjust the ticks on the color bar

	# Add labels to the color bar
	cbar.ax.set_yticklabels(["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", 
				"Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", 
				"War", "Western", "IMAX"])

	# Add text labels to each point
	#for i, label in enumerate(genre_list):
		#plt.text(x[i], y[i], label, fontsize=10, ha='center', va='bottom')

	# You can customize the appearance of the plot by adding labels, a title, grid, etc.
	plt.title('2D Embeddings Scatter Plot of Movie Genres')
	plt.xlabel('X-axis')
	plt.ylabel('Y-axis')
	
	# Show the plot
	plt.show()
