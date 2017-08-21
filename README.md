# Facebook Post Page's Cluster

### Dependences

* python (3.6)
* pandas (0.19.2)
* numpy (1.13.1)
* nltk (3.2.2)
* scipy (0.19.1)
* sklearn


### Usage

#### Download Facebook Crawler (Facebook-Page-Crawler)

`> git submodule init`

`> git submodule update`

#### Download Page Posts for clustering

`> python Crawler/Facebook_Page_Crawler.py 'SiteOmelete' '2017-07-01 00:00:00' '2017-07-30 23:59:59' --resume`

#### Clustering

`> python Cluster/run.py 'SiteOmelete'`

### Default Params

Change the information in the file `parameters.json`

```json
{
    "fb_app_id": "",
    "fb_app_secret": "",
    "pca_variance_max": 0.7,
    "range_of_cluster": [10, 50],
    "tfidf_max_features": 200000,
    "tfidf_max_df": 0.1,
    "tfidf_min_df": 0.01,  
    "tfidf_ngram_range": [1, 2]    
}
```

### Step by Step

* Data mining in text with bag of words and n-grams
* Clustering with k-means
* Automated generation of descriptors with decision tree
* Cluster engagement analysis