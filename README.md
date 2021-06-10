# M4R
M4R Project

This repository only contains the code. So training data, collected data, trained model weights etc will not be included.

**Contents**
- Data Harvesting
- - `tweet_harvester.py` Harvests tweets based on search terms or tweet ids.
- - `user_harvester.py` Harvests tweets from specific users or user account data (specifically to fill out retweet and reply networks - i.e. to be able to perform account level detection).
- - `full_text_tokeniser.py` Tokenises the tweets.
- Bot Detection Methods
- - `feature_selection.py` Performs feature selection techniques, including RFC feature importances, recursive feature elimination, and ANOVA.
- - `account_level_detection.py` Trains account level detection model: compares models and resampling techniques. Also applies account level detection model to collected datasets.
- - `tweet_level_detection.py` Trains tweet level detection: compares models.
- Application to Election Data
- - `data_exploration.py` Explores summary statistics of the Georgia and US datasets, plots distributions of datasets, compares to training dataset.
- - `sentiment_analysis.py` Performs VADER sentiment analysis: distribution and plots over time.
- - `reply_network.py` Builds simple and more complicated reply networks for the Georgia dataset.
- - `hashtag_cooccurrence_network.py` Builds a hashtag co-occurrence network that can be plotted in Gephi.