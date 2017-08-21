# Facebook Post Page's Cluster

### Dependences

* python (3.6)
* pandas (0.19.2)
* numpy (1.13.1)
* nltk (3.2.2)
* scipy (0.19.1)
* sklearn


### Usage

Update submodule

*git submodule init*
*git submodule update*

Dowlonad dataset with Post Pages
python Crawler/Facebook_Page_Crawler.py 'SiteOmelete' '2017-07-01 00:00:00' '2017-07-30 23:59:59' --resume

Clustering

python Cluster/run.py 'SiteOmelete'


### Params

Change parameters.json

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