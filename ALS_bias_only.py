# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import pickle
import gzip



data = pd.read_csv('ratings.csv')
data = data.drop(columns='timestamp')
data = data.to_numpy()


# Index the data

def get_indices():
  map_user_to_index = {}
  map_index_to_user = []
  map_movie_to_index = {}
  map_index_to_movie = []

  for user_id, movie_id, rating in data:
    if user_id not in map_user_to_index:
      index_user = len(map_index_to_user)
      map_index_to_user.append(user_id)
      map_user_to_index[user_id] = index_user
    else:
      index_user = map_user_to_index[user_id]

    if movie_id not in map_movie_to_index:
      index_movie = len(map_index_to_movie)
      map_index_to_movie.append(movie_id)
      map_movie_to_index[movie_id] = index_movie
    else:
      index_movie = map_movie_to_index[movie_id]

  return map_user_to_index, map_index_to_user, map_movie_to_index, map_index_to_movie



map_user_to_index, map_index_to_user, map_movie_to_index, map_index_to_movie = get_indices()


# Create sparse matrics
def get_sparse(data):
  data_by_user = []
  data_by_movie = []
  for i in range(len(map_user_to_index)):
    data_by_user.append([])

  for i in range(len(map_movie_to_index)):
    data_by_movie.append([])

  for user_id, movie_id, rating in data:
    index_user = map_user_to_index[user_id]
    index_movie = map_movie_to_index[movie_id]
    data_by_user[index_user].append((index_movie, rating))
    data_by_movie[index_movie].append((index_user, rating))
  return data_by_user, data_by_movie


user_train, movie_train =  get_sparse(data)

data_dict = {"user_train": user_train, "movie_train": movie_train}

with gzip.open('datas_bias.pkl', 'wb') as f:
    pickle.dump(data_dict, f)


with gzip.open('datas_bias.pkl', 'rb') as f:
    test = pickle.load(f)


user_train = test['user_train']
movie_train  = test['movie_train']


# import libraries for parallization
from tqdm import tqdm as tq
from concurrent.futures import ThreadPoolExecutor, wait


class Model_bias_only:
    """ ALS Model implementation only for user biases and item biases"""
    def __init__(self, data, lamda, gamma):
        user_train, movie_train = data
        self.user_train = user_train
        self.movie_train = movie_train
        self.lamda = lamda
        self.gamma = gamma
        self.num_user = len(user_train)
        self.num_item = len(movie_train)
        self.user_biases = np.zeros(self.num_user)
        self.item_biases = np.zeros(self.num_item)
        self.cost_train = []
        self.rmse_train= []


    def update_user_bias(self, range):
        """Update user_bias """
        for m in range:
            bias = 0
            item_counter = 0
            for n, r in self.user_train[m]:
                bias += self.lamda*(float(r) - self.item_biases[n])
                item_counter +=1

            bias = bias / (self.lamda * item_counter + self.gamma)
            self.user_biases[m] = bias

    def update_item_bias(self, range):
        """Update item_bias"""
        for n in range:
            bias = 0
            user_counter = 0
            for m, r in self.movie_train[n]:
                bias += self.lamda*(float(r) - self.user_biases[m])
                user_counter +=1

            bias = bias / (self.lamda * user_counter + self.gamma)
            self.item_biases[n] = bias
            

    def train(self, epoch):

        r1 = [i for i in range(self.num_user)]
        r2 = [i for i in range(self.num_item)]

        for _ in tq(range(epoch)):

            with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                job1 = executor.submit(self.update_user_bias, r1)
                #wait([job1])
                job2 = executor.submit(self.update_item_bias, r2)

            loss, rms = self.get_loss_rmse(self.user_train)
            self.cost_train.append(loss)
            self.rmse_train.append(rms)

        self.plot_cost()
        self.plot_RMSE()


    def plot_cost(self):
        fig, axs = plt.subplots(figsize=(5, 4))
        x = [i for i in range(len(self.cost_train))]
        y = [-i for i in self.cost_train]

        axs.plot(x, y)
        axs.set_xlabel('# of iterations, lambda = {}, gamma  = {}'.format(self.lamda, self.gamma ))
        axs.set_ylabel('Cost function')
        axs.grid(True, linestyle='--', linewidth=0.5)

    def plot_RMSE(self):
        fig, axs = plt.subplots(figsize=(5, 4))
        x = [i for i in range(len(self.rmse_train))]
        y = [i for i in self.rmse_train]

        axs.plot(x, y)
        axs.set_xlabel('# of iterations, lambda = {}, gamma  = {}'.format(self.lamda, self.gamma))
        axs.set_ylabel('RMSE')
        axs.grid(True, linestyle='--', linewidth=0.5)


    def get_loss_rmse(self, data):
        loss = 0
        sum_ = 0
        reg = 0
        RMSE_ = 0
        RMSE_counter = 0

        for m in range(self.num_user):
            for n, r in data[m]:
                err_sq = (float(r) - (self.user_biases[m] + self.item_biases[n]))**2
                sum_ += (-self.lamda/2)*err_sq
                RMSE_ += err_sq
                RMSE_counter +=1
        RMSE_total = np.sqrt(RMSE_/RMSE_counter)

        reg = (-self.gamma/2)*(np.dot(self.user_biases, self.user_biases.T) + np.dot(self.item_biases, self.item_biases.T))

        loss = sum_ + reg
        return loss, RMSE_total
