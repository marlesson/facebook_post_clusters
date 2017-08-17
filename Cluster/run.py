import argparse, sys

# data analysis and wrangling
import pandas as pd
import numpy as np

# Machine Learning
from sklearn.cluster import k_means_
from sklearn.metrics import silhouette_score
from sklearn import tree
from sklearn.decomposition import PCA

# load nltk's SnowballStemmer as variabled 'stemmer'
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import ward, dendrogram
#Natual Language
import nltk 

from scipy import io

# Util
from util import *
from graphics import *


##########################################################################################################
if __name__ == '__main__':

    # Set crawler target and parameters.
    parser = argparse.ArgumentParser()

    parser.add_argument("target", help="Set page")

    args = parser.parse_args()

    target = str(args.target)

    # ------------------------------------------------
    # Basic Parameters
    # ------------------------------------------------
    page             = target

    outpath          = 'Result/'+page+'/'
    range_of_cluster = range(10, 15)
    pca_variance_max = 0.80
    f_stopwords      = 'Cluster/stopwords.txt'



    # ----------------------------------------------
    # Read json's posts for DataFrame 
    # ----------------------------------------------
    print(target)
    print("------------ %s ------------\n" %page)

    print("\n\n---------------------------------------")
    print(">> Read json's posts for DataFrame ")
    print("---------------------------------------")
    posts   = post_json_to_list(outpath)
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
    print("Statistics of Engagement: ")
    mean, std = sample_statistic(df['eng'])
    ds = df[df['eng'] <= mean*2*(std)]
    print("Remove", len(df)-len(ds), "of a total of", len(df))
    df = ds
    print()

    # Show statistics after remove outliers
    print()
    print("Statistics of Engagement: ")
    plot_dist_prob(outpath, df['eng'])
    sample_statistic(df['eng']) 
    print()


    # ----------------------------------------------
    # 
    # ----------------------------------------------
    print("\n\n---------------------------------------")
    print(">> Text Mining  ")
    print("---------------------------------------")

    # Create new column with all columns text
    df['text'] = df['title']+" "+df['description']+" "+df['message']
    df['text'] = df['text'].apply(remove_accentuation)

    ### Stopwords, stemming, and tokenizing
    stopwords = nltk.corpus.stopwords.words('portuguese')
    with open(f_stopwords, 'r')  as f:
      stopwords_ext = [x.strip('\n') for x in f.readlines()]

    stopwords.extend(stopwords_ext)
    stopwords = [remove_accentuation(w) for w in stopwords]

    # Tokenize all text in Posts for a vocabulary
    totalvocab_tokenized = []

    for i in df['text']:
        allwords_tokenized = tokenize_only(i)
        totalvocab_tokenized.extend(allwords_tokenized)

    # Tf-idf and document similarity
    # TODO: change this params for better results
    tfidf_vectorizer = TfidfVectorizer(max_features=200000,
                                       max_df=0.7,
                                       min_df=0.01,  stop_words=stopwords, 
                                       use_idf=True, tokenizer=tokenize_only, 
                                       ngram_range=(1,3))

    tfidf_matrix = tfidf_vectorizer.fit_transform(df.text) #fit the vectorizer to synopses
    terms = tfidf_vectorizer.get_feature_names()

    print("Total features(tokens) %s" %tfidf_matrix.shape[1])
    print()


    # ----------------------------------------------
    # PCA - Dimensionality reduction
    # ----------------------------------------------
    print("\n\n---------------------------------------")
    print(">> PCA - Dimensionality reduction ")
    print("---------------------------------------")
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
    print("\n\n---------------------------------------")
    print(">> K-means executing...")
    print("---------------------------------------")
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

    plot_silhouette(outpath, clusters, AvgSil)

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

    plot_total_posts_per_cluster(outpath, df)
    plot_avg_eng_per_cluster(outpath, df)
    plot_box_avgt_eng_per_cluster(outpath, df)
    plot_dist_prob_per_cluster(outpath, df)


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

        # ----------------------------------------------
        # Desision Tree for build description of cluster
        # ----------------------------------------------
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


    # ----------------------------------------------
    # Export Results
    # ----------------------------------------------    
    print("\n\n---------------------------------------")
    print(">> Export Results")
    print("---------------------------------------")
    df_cluster = pd.DataFrame(data_cluster, columns=['cluster', 'topics', 'count_posts', 'total_eng',
                                            'mean_eng', 'min_eng', 'max_eng', 'median_eng'])

    print("Export: "+outpath+'resume_clusters.csv')
    df_cluster.sort_values(['mean_eng'], ascending=[0]).to_csv(outpath+'resume_clusters.csv', sep=';')

    print("Export: "+outpath+'posts.csv')
    df[['id', 'title', 'created_time', 'eng', 'cluster']].sort_values(['cluster']).to_csv(outpath+'posts.csv', sep=';')