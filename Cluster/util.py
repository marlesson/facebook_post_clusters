import numpy as np
import scipy.stats as stats
import os.path
import json
import pandas as pd
import math

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import k_means_
from sklearn.metrics import silhouette_score

# load nltk's SnowballStemmer as variabled 'stemmer'
from unicodedata import normalize
import nltk #Natual Language
import re

def sample_statistic(sample):
    scatter,loc,mean = stats.lognorm.fit(sample,floc=0) #Gives the paramters of the fit 
    var = math.e**(scatter**2) # Variancia no normal 
    median = np.median(sample)
    x_fit = np.linspace(sample.min(),sample.max(),100)
    pdf_fitted = stats.lognorm.pdf(x_fit,scatter,loc,mean) #Gives the PDF

    print("variance for data is %s" %var)
    print("mean of data is %s" %mean)    
    print("mean of data is %s (lognormal)" %np.mean(sample))
    print("median of data is %s" %median)
    return (mean, var)


def remove_accentuation(txt):
    return normalize('NFKD', txt).encode('ASCII','ignore').decode('ASCII')

def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            if len(token) > 1:
                filtered_tokens.append(token) 
    return filtered_tokens

# Clustering
def dist(X, Y = None):
    # if Y == None:
    #   return euclidean_distances(X) 
    # return euclidean_distances(X, Y)
    if Y == None:
      return 1-cosine_similarity(X) 
    return 1-cosine_similarity(X, Y)

def clustering(nclust, sparse_data):
    print("Cluster", nclust)
    # Manually override euclidean
    def euc_dist(X, Y = None, Y_norm_squared = None, squared = False):
        return dist(X,Y)
    k_means_.euclidean_distances = euc_dist

    kmeans = k_means_.KMeans(n_clusters = nclust)
    _ = kmeans.fit(sparse_data)
    return kmeans, _
    
# Transforma matriz em binária
def to_binaryMatrix(x):
    return 1 if x > 0 else 0

to_binaryMatrix = np.vectorize(to_binaryMatrix)


def engagement(post):
    return post['comments_count']+post['shares_count']+post['total']

def post_json_to_list(path):
    json_lst = []
    # Ler arquivos
    feeds = open(path+'feed_ids', 'r')
    for line in feeds.readlines():
        id_post = line.rstrip()
        file_name = path+id_post+'/'+id_post+'.json'
        if os.path.isfile(file_name):
            post = open(file_name, 'r').read()
            post = json.loads(post)
            json_lst.append([post['id'],
                            post['message'],
                            post['link'],
                            pd.to_datetime(post['created_time']),
                            post['comments_count'],
                            post['shares_count'],
                            post['title'],
                            post['description'],
                            post['target'],
                            post['like'],
                            post['love'],
                            post['haha'],
                            post['wow'],
                            post['sad'],
                            post['angry'],
                            post['total'],
                            engagement(post)])
    return  json_lst
