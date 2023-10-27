# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import pickle
import gzip


# Shuffle and split data
def structure_data(path):
    data = pd.read_csv(path)
    data = data.drop(columns='timestamp')
    data = data.to_numpy()    
    np.random.shuffle(data)
    train_data = np.split(data, [0,(data.shape[0]*80//100)])[1]
    test_data = np.split(data, [0,(data.shape[0]*80//100)])[2]
    return data, train_data, test_data



data, train_data, test_data = structure_data('ratings.csv') 

# Index data
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

# Get sparse matrix
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


user_train, movie_train =  get_sparse(train_data)
user_test, movie_test =  get_sparse(test_data)

# We use pickle to serialize the data
with gzip.open('datas.pkl', 'wb') as f:
    pickle.dump(data_dict, f)


with gzip.open('datas.pkl', 'rb') as f:
    test = pickle.load(f)

data_dict = {"user_train": user_train, "movie_train": movie_train, "user_test": user_test}
user_train = test['user_train']
user_test  = test['user_test']
movie_train  = test['movie_train']

# We use concurrent.futures to make the algorithm run parallel
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
        self.plot_cost()
        self.plot_RMSE()
        
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
