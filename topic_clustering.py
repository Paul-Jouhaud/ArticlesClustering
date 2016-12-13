from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


"""
    TOPIC CLUSTERING:
====

    1. Vectorize the content of every articles
    2. Clusterize all articles by their topic using KMeans
        - : We must define K, the number of cluster we wants
        + : It's faster than DBScan
        + : The number of topics is easier to evaluate than the number of events in a given feed
    3. Add every articles from the feed and the knowledge board in their cluster

====

    k           : number of topics
    corpus      : list of every articles
    feed        : feed of articles
    kb_articles : list of articles in the knowledge board

"""
def topic_clustering(k, corpus, feed, kb_articles):
    # Vectorize everything and clusterize by topics
    vectorizer = TfidfVectorizer(stop_words='english')
    corpus_vectorized = vectorizer.fit_transform(corpus)
    model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
    model.fit(corpus_vectorized)
    # Uncomment to print the most important terms for the current topic
    # print ("Top terms per clusters:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    clusters = []
    for i in range(k):
        cluster = {}
        topics = []
        # Uncomment to print the most important terms for the current topic
        # print ("Cluster %d:" % i)
        for ind in order_centroids[i, :3]:
            if terms[ind].isnumeric():
                ind = order_centroids[i, 4]
            # Uncomment to print the most important terms for the current topic
            # print(" %s" % terms[ind])
            topics.append(terms[ind])
        cluster['topics'] = topics
        clusters.append(cluster)

    # Add articles from the feed in their cluster
    len_feed = len(feed.get('items'))
    for i in range(len_feed):
        prediction = model.predict(corpus_vectorized[i])
        articles = clusters[prediction].get('articles', [])
        articles.append(feed.get('items')[i])
        clusters[prediction]['articles'] = articles
    # Add saved articles from the knowledge board in their cluster
    for i in range(len(kb_articles)):
        prediction = model.predict(corpus_vectorized[i + len_feed])
        saved_articles = clusters[prediction].get('saved_articles', [])
        saved_articles.append(kb_articles[i])
        clusters[prediction]['saved_articles'] = saved_articles
    return clusters