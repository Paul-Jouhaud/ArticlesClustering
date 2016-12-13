import json
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re


"""
    EVENT CLUSTERING:
====

    1. Standardize every number in the content of the articles
    2. Vectorize the content of every articles
        - We set max_df at a low value, so that too common terms in the cluster are not analyzed
    3. Clusterize all articles by their topics using DBSCan
        + : We don't have to set number of cluster we want
        + : DBScan is theorically able to find every events
        ~ : DBScan is slower than KMeans, but as we use it on smaller datasets, it's still runnable on a laptop
    4. Send every events back to the cluster

====

    articles: all articles from the cluster

"""
def event_clustering(articles):
    corpus = []
    for article in articles:
        content = article.get('content')
        expression = r"\d{1,3}([' ',',']\d{1,3})+"
        if re.search(expression, content) is not None:
            number = re.search(expression, content).group(0)
            newNumber = number.replace(' ', '')
            newNumber = newNumber.replace(',', '')
            content = content.replace(number, newNumber)
        corpus.append(content)
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.3, min_df=0)
    X = vectorizer.fit_transform(corpus)
    dbscan = DBSCAN(eps=0.8, min_samples=1)
    events = dbscan.fit_predict(X)
    core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True
    labels = dbscan.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    events = []
    for i in range(n_clusters_):
        events.append([])
    for i in range(len(articles)):
        event = labels[i]
        events[event].append(articles[i])
    return events