import argparse, sys

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import json
import os.path
import math

# visualization
from wordcloud import WordCloud
from scipy.stats import lognorm
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.cluster import k_means_
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn import tree
from sklearn.decomposition import PCA

# load nltk's SnowballStemmer as variabled 'stemmer'
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import ward, dendrogram
from unicodedata import normalize
import nltk #Natual Language
import re
from scipy import sparse, io
import pickle


# Functions
# ------------ Datasience--------------
# Distribuição das Curtidas
def plot_dist_prob(samples):
    N_bins = 50

    # make a fit to the samples
    shape, loc, scale = stats.lognorm.fit(samples, floc=0)
    x_fit       = np.linspace(samples.min(), samples.max(), 100)
    samples_fit = stats.lognorm.pdf(x_fit, shape, loc=loc, scale=scale)

    # plot a histrogram with linear x-axis
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5), gridspec_kw={'wspace':0.2})
    counts, bin_edges, ignored = ax1.hist(samples, N_bins, histtype='stepfilled', alpha=0.4,
                                          label='histogram')

    # calculate area of histogram (area under PDF should be 1)
    area_hist = ((bin_edges[1:] - bin_edges[:-1]) * counts).sum()

    # plot fit into histogram
    ax1.plot(x_fit, samples_fit*area_hist, label='PDF log-normal', linewidth=2)
    ax1.legend()
    plt.ylabel('Total posts')
    plt.xlabel('Engagement')   

    # equally sized bins in log10-scale and centers
    bins_log10 = np.logspace(np.log10(samples.min()), np.log10(samples.max()), N_bins)
    bins_log10_cntr = (bins_log10[1:] + bins_log10[:-1]) / 2

    # histogram plot
    counts, bin_edges, ignored = ax2.hist(samples, bins_log10, histtype='stepfilled', alpha=0.4,
                                          label='histogram')

    # calculate length of each bin and its centers(required for scaling PDF to histogram)
    bins_log_len = np.r_[bin_edges[1:] - bin_edges[: -1], 0]
    bins_log_cntr = bin_edges[1:] - bin_edges[:-1]

    # get pdf-values for same intervals as histogram
    samples_fit_log = stats.lognorm.pdf(bins_log10, shape, loc=loc, scale=scale)
    
    # pdf-values for centered scale
    samples_fit_log_cntr = stats.lognorm.pdf(bins_log10_cntr, shape, loc=loc, scale=scale)

    # plot fitted and scaled PDFs into histogram
    ax2.plot(bins_log10_cntr, 
         samples_fit_log_cntr * bins_log_cntr * counts.sum(), '-', 
         label='PDF normal', linewidth=2)


    ax2.set_xscale('log')
    ax2.set_xlim(bin_edges.min(), bin_edges.max())
    ax2.legend(loc=3)
    plt.savefig(outpath+'probability_dist.png')
    plt.clf()
       

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

#---------Text Mining -------------
def remover_acentos(txt):
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


def plot_silhouette(values):
    # Var IntraGrupo
    fig, ax = plt.subplots(figsize=(15,6), facecolor='w')
    ax.yaxis.grid() # horizontal lines

    plt.plot(clusters, AvgSil)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average silhouette ')
    plt.savefig(outpath+'silhouette_cluster.png')
    plt.clf()

def plot_total_posts_per_cluster(df):
    # Total de Posts por Cluster
    count = df[['cluster', 'eng']].groupby(['cluster'], as_index=False).count()

    # plot
    fig, ax = plt.subplots(figsize=(15,6), facecolor='w')
    ax.yaxis.grid() # horizontal lines

    plt.xlabel('Cluster')
    plt.ylabel('Total Posts')
    plt.bar(count.cluster, count.eng)
    plt.xticks(range(len(count.cluster)), count.cluster)
    plt.savefig(outpath+'plot_total_posts_per_cluster.png')
    plt.clf()

def plot_avg_eng_per_cluster(df):
    # Média do engajamento por cluster
    average = df[['cluster', 'eng']].groupby(['cluster'], as_index=False).mean()

    
    # plot
    fig, ax = plt.subplots(figsize=(15,6), facecolor='w')
    ax.yaxis.grid() # horizontal lines

    plt.xlabel('Cluster')
    plt.ylabel('Average engagement')
    plt.bar(average.cluster, average.eng)
    plt.xticks(range(len(average.cluster)), average.cluster)  
    plt.savefig(outpath+'plot_avg_eng_per_cluster.png')
    plt.clf()

def plot_box_avgt_eng_per_cluster(df):
    # Média do engajamento por cluster
    average = df[['cluster', 'eng']].groupby(['cluster'], as_index=False).mean()

    # Plot Box
    data_plot_log = []
    data_plot = []
    
    for k in range(0, average.cluster.size):
        data_plot_log.append(np.log10(df[df['cluster'] == k].eng))
        data_plot.append(df[df['cluster'] == k].eng)

    # Create a figure instance
    fig = plt.figure(1, figsize=(15, 6))
    
    plt.xlabel('Cluster')
    plt.ylabel('Engagement')

    # Create an axes instance
    ax = fig.add_subplot(111)
    #ax.yaxis.grid() # horizontal lines

    # Create the boxplot
    bp = ax.boxplot(data_plot, showmeans=True, meanline=True, showfliers=False)
    ax.set_xticklabels(average.cluster)
    plt.savefig(outpath+'plot_box_avgt_eng_per_cluster.png')
    plt.clf()

def plot_dist_prob_per_cluster(df):
    g = sns.FacetGrid(df, col='cluster', col_wrap=4)
    g.map(plt.hist, 'eng', bins=20)
    g.set_axis_labels("Posts", "Engagement")
    g.savefig(outpath+'plot_dist_prob_per_cluster.png')

def plot_word_cloud(cluster, text):
    wordcloud = WordCloud(background_color="white", width = 1000, height = 500).generate(text)
    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig(outpath+'wordsclound/%s_plot_word_cloud.png' %cluster)
    plt.clf()
    
# Transforma matriz em binária
def to_binaryMatrix(x):
    return 1 if x > 0 else 0

to_binaryMatrix = np.vectorize(to_binaryMatrix)

# -----------------------

# Funcoes
def engajamento(post):
    return post['comments_count']+post['shares_count']+post['total']

def read_posts(path):
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
                            engajamento(post)])
    return  json_lst

##########################################################################################################
if __name__ == '__main__':
    # Set crawler target and parameters.
    parser = argparse.ArgumentParser()

    parser.add_argument("target", help="Set page")

    args = parser.parse_args()

    target = str(args.target)
    # Parametros
    page        = target

    outpath     = 'Result/'+page+'/'
    range_of_cluster = range(10, 15)
    pca_variance_max = 0.40
    f_stopwords = 'Cluster/stopwords.txt'

    # ----------------------------------------------
    # Ler os dados
    # ----------------------------------------------
    print(target)
    print("------------ %s ------------\n" %page)


    # Ler dos dados
    posts   = read_posts(outpath)
    df      = pd.DataFrame(posts, columns=['id', 'message', 'link', 'created_time',
                                              'comments_count','shares_count', 'title','description',
                                              'target','like','love','haha','wow',
                                              'sad','angry', 'total', 'eng'])
    
    # Create new coluns
    df['ct_month']   = df['created_time'].dt.month
    df['ct_weekday'] = df['created_time'].dt.dayofweek
    df['ct_hour']    = df['created_time'].dt.hour

    print('Total posts:', df.shape[0])

    # Remove outliers, values of engajment higher 2 ds
    mean, std = sample_statistic(df['eng'])
    ds = df[df['eng'] <= mean*2*(std)]
    print("Remove", len(df)-len(ds), "of a total of", len(df))
    df = ds

    # ----------------------------------------------
    # Datascience 
    # ----------------------------------------------
    print()
    print("Statistics of Engagement: ")
    plot_dist_prob(df['eng'])
    sample_statistic(df['eng']) 
    print()

    # ----------------------------------------------
    # Text Mining - Cria a string 
    # ----------------------------------------------

    # Create new column with all columns text
    df['text'] = df['title']+" "+df['description']+" "+df['message']
    df['text'] = df['text'].apply(remover_acentos)

    ### Stopwords, stemming, and tokenizing
    stopwords = nltk.corpus.stopwords.words('portuguese')
    with open(f_stopwords, 'r')  as f:
      stopwords_ext = [x.strip('\n') for x in f.readlines()]

    stopwords.extend(stopwords_ext)
    stopwords = [remover_acentos(w) for w in stopwords]

    # Tokenize all text in Posts for a vocabulary
    totalvocab_tokenized = []

    for i in df['text']:
        allwords_tokenized = tokenize_only(i)
        totalvocab_tokenized.extend(allwords_tokenized)

    # Tf-idf and document similarity
    tfidf_vectorizer = TfidfVectorizer(max_features=200000,
                                       max_df=0.7,
                                       min_df=0.01, stop_words=stopwords, 
                                       use_idf=True, tokenizer=tokenize_only, 
                                       ngram_range=(1,3))

    tfidf_matrix = tfidf_vectorizer.fit_transform(df.text) #fit the vectorizer to synopses
    terms = tfidf_vectorizer.get_feature_names()

    print("Total features(tokens) %s" %tfidf_matrix.shape[1])
    print()


    # ----------------------------------------------
    # PCA
    # ----------------------------------------------
    print("PCA")
    pca = PCA()
    
    pca_fit = pca.fit_transform(tfidf_matrix.toarray())
    print("Explained Varicante:", pca.explained_variance_.sum()) 

    # PCA with Explained variance > 80%
    s = pca.explained_variance_ratio_
    sum=0.0
    comp=0

    for _ in s:
        sum += _
        comp += 1
        if(sum>=pca_variance_max):
            break

    print("Componentes: ", comp)
    
    pca = PCA(n_components=comp)
    pca_fit = pca.fit_transform(tfidf_matrix.toarray())
    print("Explained Variance:", pca.explained_variance_ratio_.sum()) 

    # Create dataframe with PCA components

    df_pca = pd.DataFrame(pca_fit)
    df_pca.index = df.index

    # ----------------------------------------------
    # k-means clustering
    # ----------------------------------------------
    print("k-menas executing")
    clusters=range_of_cluster
    varInterGrupo=[]
    varIntraGrupo=[]
    AvgSil = []

    ma_attr   = tfidf_matrix.toarray()
    #ma_attr   = df_pca
    _dist     = dist(df_pca)

    for total_k in clusters:
        # k-means
        km, fit = clustering(total_k, df_pca)

        # http://www.sthda.com/english/wiki/determining-the-optimal-number-of-clusters-3-must-known-methods-unsupervised-machine-learning   
        # Average silhouette 
        avg_sil = silhouette_score(_dist, km.labels_, metric="precomputed")
        AvgSil.append(avg_sil)

    plot_silhouette(AvgSil)

    # Parametros de K ideal
    num_clusters =  range_of_cluster[np.argmax(AvgSil)]
    #num_clusters = 30
    print()
    print("Choice %s clusters" %num_clusters)
    # k-means
    km, _ = clustering(num_clusters, df_pca)

    # Add column cluster
    df['cluster'] = km.labels_.tolist()

    # Add distance of group center
    dists = []
    for i, k in enumerate(km.labels_):
        cluster_center = km.cluster_centers_[k]
        dists.append(dist([df_pca.iloc[i]],[cluster_center]).reshape(1)[0])
    df['dist'] = dists

    plot_total_posts_per_cluster(df)
    plot_avg_eng_per_cluster(df)
    plot_box_avgt_eng_per_cluster(df)
    plot_dist_prob_per_cluster(df)
    # ----------------------------------------------
    # Per Cluster
    # ----------------------------------------------

    print("Top terms per cluster:")
    print()

    # Ordena pelo indice com maior quantidade, retorna os indices
    # 
    order_centroids    = km.cluster_centers_.argsort()[:, ::-1] 
    binary_matrix_attr = to_binaryMatrix(ma_attr)
    data_cluster       = []

    for i in range(num_clusters):
        print("Cluster %d words: " % i, end='')
        
        # Keywords frequency
        keywords = []
        for ind in order_centroids[i, :20]: #replace 6 with n words per cluster
            keywords.append(terms[ind])
        
        df_keyword    = pd.DataFrame({'keyword': keywords})
        count_keyword = df_keyword['keyword'].value_counts()
        print(', '.join(np.array(count_keyword.keys())[0:10].tolist()))

        # Keywors Tree
        cluster_bool = [1 if x == i else 0 for x in df.cluster.values]
        clf = tree.DecisionTreeClassifier(criterion='entropy')
        clf = clf.fit(binary_matrix_attr, cluster_bool)

        importances = clf.feature_importances_
        indices     = np.argsort(importances)[::-1]

        print("Feature ranking:")
        keywords_importance = []

        for f in range(5):
            if importances[indices[f]] > 0.01:
                keywords_importance.append(terms[indices[f]])
                print("\t%d. %s (%f)" % (f + 1, terms[indices[f]], importances[indices[f]]))
        
        for index in df[df['cluster'] == i].sort_values(['dist'], ascending=[True])[0:5].index.values:
            data = df.loc[index]
            print("Title:", data.title, " | ", data.id)    
        
        print() #add whitespace
        print() #add whitespace
        
        # Statistics
        df_cluster  = df[df['cluster'] == i]
        cluster     = i
        topics      = ', '.join(keywords_importance)
        count_posts = len(df_cluster)
        total_eng   = df_cluster.eng.sum()
        mean_eng    = df_cluster.eng.mean()
        min_eng     = df_cluster.eng.min()
        max_eng     = df_cluster.eng.max()
        median_eng  = df_cluster.eng.median() 

        data_cluster.append([cluster, topics, count_posts, total_eng, mean_eng, min_eng, max_eng, median_eng])

    df_cluster = pd.DataFrame(data_cluster, columns=['cluster', 'topics', 'count_posts', 'total_eng',
                                            'mean_eng', 'min_eng', 'max_eng', 'median_eng'])

    df_cluster.sort_values(['mean_eng'], ascending=[0]).to_csv(outpath+'resume.csv', sep=';')
