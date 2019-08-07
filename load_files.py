# Load libraries
import pandas
import numpy as np
import pdb


def load_dataset():
 url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
 names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
 dataset = (pandas.read_csv(url, names=names)).to_numpy()
 np.random.shuffle(dataset)
 input,output =  dataset[:,:4],dataset[:,-1:]
 train_data_x,test_data_x,train_output_y,test_output_y = input[:140],input[-10:],output[:140],output[-10:]
 pdb.set_trace()
 return  train_data_x,test_data_x,train_output_y,test_output_y

load_dataset()
