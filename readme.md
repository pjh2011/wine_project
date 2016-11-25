
## Project Summary

This project attempts to find similar wines based on user-written "tasting notes" from the website CellarTracker. Tasting notes are similar to wine reviews, but the idea is for each user to describe their experience drinking the wine and try to verbalize what they thought the the wine tasted like. Using TFIDF and Word2Vec I project the tasting notes written for each wine into a vector, and can then compare wine similarities based on the cosine distance between vectors. I used ~730 thousand wine tasting notes written for ~30,800 wines scraped from the website cellartracker.com.

## Data Collection

I used ParseHub's web scraping platform to scrape data from cellartracker.com, then used Python's Pandas package for data cleaning and storage.

## Modeling

I started with TFIDF analysis of the wine tasting notes. I found that many people were mentioning varietal and region names in their tasting notes contributing to some data leakage. Using K-means clustering I found that wines were clustering very neatly along their varietal types. For example most of the Pinot Noirs ended up in one cluster and most of the Cabernets in another cluster. After removing stopwords specific to wine eg. "Kirsch", "Crus", "Cab", and "Julien", performed singular value decomposition (SVD) and non-negative matrix factorization (NMF) to inspect the numerical representations of the wine reviews. With a lack of "ground truth" for wine similarity I relied on unsupervised learning techniques to verify that my results were logical and matched real-world experience.

Hypothesis: although not a perfect metric, different varietals should generally have distinguishable taste and therefore distinguishable tasting note language. Varietals that taste very different eg. Chardonnay and Merlot should be less similar than varietals that are more substitutable eg. Merlot and Cabernet. SVD verified this, as white wines and red wines separated into two clear clusters in just the first two principal components.

Used NMF to model latent topics in the wine tasting notes

## Adding in Word Vectors
