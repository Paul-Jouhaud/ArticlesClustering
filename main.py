import json
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from math import floor
from random import random
from datetime import datetime
from topic_clustering import topic_clustering
from event_clustering import event_clustering

"""
    CREATE NEW FEED:
====

    1. Load the whole feed
    2. Loads every knowledge boards given in kbs
    3. Send every articles to topic_clustering and generate clusters of topics
    5. Send every cluster to event_clustering and generate sub-clusters of events
    6. Generate a feed

====

    k                               : number of topics
    feed_json                       : path to the feed
    max_new_feed_length             : max size of the feed if content_of_feed_proportional_kb is True
    content_of_feed_proportional_kb : define if the number of articles from each cluster is proportional to the articles
                                      in the knowledge board

"""
def create_new_feed(k, feed_json, kbs, max_new_feed_length, content_of_feed_proportional_kb):
    # Create datasets to analyze : feed + knowledge boards
    t0 = time.time()
    feed = json.loads(open(feed_json, "r").read())
    corpus = []
    for article in feed.get('items'):
        corpus.append(article.get('content'))
    kb_articles = []
    nb_kb_articles = 0
    for kb_json in kbs:
        kb = json.loads(open(kb_json, "r").read())
        nb_kb_articles = nb_kb_articles + len(kb.get('items'))
        for article in kb.get('items'):
            kb_articles.append(article.get('content'))
            corpus.append(article.get('content'))

    # End of the generation of the corpus
    t1 = time.time()

    # Generate the list of clusters
    clusters = topic_clustering(k=k, corpus=corpus, feed=feed, kb_articles=kb_articles)

    # End of the topic clustering
    t2 = time.time()

    # Clusterize by events for each cluster
    for index in range(len(clusters)):
        currentCluster = clusters[index]
        events = event_clustering(currentCluster.get('articles'))
        clusters[index]['events'] = events

    # End of the event clustering
    t3 = time.time()

    # Print events in cluster 4
    """
    events = clusters[4].get('events')
    countL10A = 0
    countM10A = 0
    countL10C = 0
    countM10C = 0
    for event in events:
        if len(event) < 10:
            countL10C += 1
        else:
            countM10C += 1
        print "==== New Event ===="
        for article in event:
            if len(event) < 10:
                countL10A += 1
            else:
                countM10A += 1
            print article.get('content')
    print("\n\n==== Additional Informations ====")
    print ("Number of cluster of size > 10 articles:")
    print (countM10C)
    print ("Number of cluster of size < 10 articles:")
    print (countL10C)
    print ("Number of articles in cluster of size > 10 articles:")
    print (countM10A)
    print ("Number of articles in cluster of size < 10 articles:")
    print (countL10A)
    """

    # Define how many articles we need for each topic
    articles_needed = []
    for cluster in clusters:
        saved_articles = cluster.get('saved_articles', [])
        number_of_articles = len(saved_articles)
        if content_of_feed_proportional_kb:
            number_of_articles_to_add = int(floor(number_of_articles * max_new_feed_length / nb_kb_articles))
        else:
            if number_of_articles > 0:
                number_of_articles_to_add = 5
            else:
                number_of_articles_to_add = 0
        articles_needed.append(number_of_articles_to_add)

    # Create the new feed
    new_feed = populate_new_feed(clusters=clusters, articles_needed=articles_needed)
    t4 = time.time()

    # Print usefull informations
    print("\n\n==== New Feed Informations ====")
    print("Number of articles in the new feed :")
    print(len(new_feed))
    print("\n\n==== Runtime Informations ====")
    print("Generation of the corpus of articles:" + str(t1-t0) + "s")
    print("Topic Clustering runtime:" + str(t2-t1) + "s")
    print("Event Clustering runtime: " + str(t3-t2) + "s")
    print("New feed generation: " + str(t4-t3) + "s")
    print("Total runtime: " + str(t4-t0) + "s")

    return new_feed


"""
    POPULATE NEW FEED:
====

    1. Add the required number of articles in the new feed
    2. Randomize the selection of articles so that it's not the same feed if re-generated

====

    clusters        : all clusters generated previously
    articles_needed : array that contains the number of articles we want for each topic

"""
def populate_new_feed(clusters, articles_needed):
    new_feed = []
    for i in range(len(articles_needed)):
        for nb_articles in range(articles_needed[i]):
            events = clusters[i].get('events')
            index = int(floor(random() * len(events)))
            event = events[index]
            last_article = event[0]
            for article in event:
                if(article.get('published') > last_article.get('published')):
                    last_article = article
            new_feed.append(article)
    return new_feed


"""
    PRINT FEED:

    1. Print the number of articles in the feed
    2. Print the content of every articles in the feed

"""
def print_feed(feed):
    print("\n\n==== New Feed ====")
    print("Number of articles in the new feed :")
    print(len(feed))
    for article in feed:
        print(u"{} : {}".format(datetime.utcfromtimestamp(article.get('published') / 1000).strftime(u'%m/%d/%Y %I:%M %p'), article.get('content')).encode("utf-8"))


"""
    GENERATE 3 DIFFERENT FEEDS:

    1. Generate 3 feeds, each using a single knowledge board
    2. Print these 3 feeds

"""
def generate_3_different_feeds():
    max_new_feed_length = 50
    content_of_feed_proportional_kb = True
    kb0 = ['firstKnowledgeBoard']
    kb1 = ['secondKnowledgeBoard']
    kb2 = ['thirdKnowledgeBoard']
    feed0 = create_new_feed(k=20, feed_json="'feedOfArticles'", kbs=kb0,
                            max_new_feed_length=max_new_feed_length,
                            content_of_feed_proportional_kb=content_of_feed_proportional_kb)
    feed1 = create_new_feed(k=20, feed_json="feedOfArticles", kbs=kb1,
                            max_new_feed_length=max_new_feed_length,
                            content_of_feed_proportional_kb=content_of_feed_proportional_kb)
    feed2 = create_new_feed(k=20, feed_json="feedOfArticles", kbs=kb2,
                            max_new_feed_length=max_new_feed_length,
                            content_of_feed_proportional_kb=content_of_feed_proportional_kb)
    print_feed(feed=feed0)
    print_feed(feed=feed1)
    print_feed(feed=feed2)


"""
    GENERATE A SINGLE FEED:

    1. Generate a single feed, using every knowledge board we have
    2. Print this new feed

"""
def generate_single_feed():
    max_new_feed_length = 50
    content_of_feed_proportional_kb = True
    kb = ['firstKnowledgeBoard']
    kb.append('secondKnowledgeBoard')
    kb.append('thirdKnowledgeBoard')
    feed = create_new_feed(k=20, feed_json="feedOfArticles", kbs=kb,
                           max_new_feed_length=max_new_feed_length,
                           content_of_feed_proportional_kb=content_of_feed_proportional_kb)
    print_feed(feed=feed)


if __name__ == '__main__':
    # Comment if you don't want to print a single feed
    generate_single_feed()
    # Uncomment if you want to print 3 different feeds
    # generate_3_different_feeds
