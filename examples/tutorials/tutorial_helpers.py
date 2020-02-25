import numpy as np
from numpy.random import seed, rand
import itertools

def exponential_family (lam, y, theta, theta_y):
    # without normalization
    return np.exp(theta_y * y + y * np.dot(theta, lam))

# create vector describing cumulative distribution of lambda_1, ... lambda_m, Y
def make_pdf(m, v, theta, theta_y, lst):
    p = np.zeros(len(lst))
    for i in range(len(lst)):
        labels = lst[i] 
        p[i] = exponential_family(labels[0:m], labels[v-1], theta, theta_y)
        
    return p/sum(p)

def make_cdf(pdf):
    return np.cumsum(pdf)

# draw a set of lambda_1, ... lambda_m, Y based on the distribution
def sample(lst, cdf):
    r = np.random.random_sample()
    smaller = np.where(cdf < r)[0]
    if len(smaller) == 0:
        i = 0
    else:
        i = smaller.max() + 1
    return lst[i]

def generate_data(n, theta, m, theta_y=0):
    v = m+1
    
    lst = list(map(list, itertools.product([-1, 1], repeat=v)))
    pdf = make_pdf(m, v, theta, theta_y, lst)
    cdf = make_cdf(pdf)

    sample_matrix = np.zeros((n,v))
    for i in range(n):
        sample_matrix[i,:] = sample(lst,cdf)
        
    return sample_matrix

def synthetic_data_basics():
    seed(0)
    
    n_train = 10000
    n_dev = 500
    
    m = 5
    theta = [1.5,1.5,.2,.2,.05]
    abstain_rate = [.80, .88, .28, .38, .45]
    
    train_data = generate_data(n_train, theta, m)
    dev_data = generate_data(n_dev, theta, m)
    
    L_train = train_data[:,:-1]
    L_dev = dev_data[:,:-1]
    Y_dev = dev_data[:,-1]
    
    train_values = rand(n_train * m).reshape(L_train.shape)
    dev_values = rand(n_dev * m).reshape(L_dev.shape)
    
    L_train[train_values < (abstain_rate,) * n_train] = 0
    L_dev[dev_values < (abstain_rate,) * n_dev] = 0
    
    return L_train, L_dev, Y_dev

def print_statistics(L_dev, Y_dev):
    m = L_dev.shape[1]
    
    for i in range(m):
        acc = np.sum(L_dev[:,i] == Y_dev)/np.sum(L_dev[:,i] != 0)
        abstains = np.sum(L_dev[:,i] == 0)/Y_dev.shape[0]
        
        print('LF {}: Accuracy {}%, Abstain rate {}%'.format(
            i, int(acc * 100), int((abstains) * 100)))