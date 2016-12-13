# articles_clustering

This project is a sample Machine Learning project.
We have several json inputs, a feed of articles and some articles that were saved by the user because he found them interesting.

We can divide this project in two phases :
- Clustering the data
- Create a personnalized feed

## Clustering
To find relevant articles to show the user, we first want to clusterize them to know what do they talk about. 
We first clusterize these articles by subjects/topics, then we clusterize them more precisely inside these subjects, by events.

## Generation of a personnalized feed
We use the data from the saved articles to create a smaller feed of articles, composed solely of relevant articles. Because we have also clusterized the articles by events, we can also choose not to present him with articles that are too similar.

## Requirements
You need to have sklearn installed, because we use intensively its algorithms. This project was made with Python 2.7