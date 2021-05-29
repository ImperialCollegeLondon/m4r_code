# M4R
M4R Project

This repository only contains the code. So training data, collected data, trained model weights etc will not be included.

**Contents**
- Data Harvesting
- - `tweet_harvester.py` Harvests tweets based on search terms or tweet ids
- - `user_harvester.py` Harvests tweets from specific users or user account data (specifically to fill out retweet and reply networks - i.e. to be able to perform account level detection)
- - `full_text_tokeniser.py` Tokenises the tweets
- Bot Detection Methods
- - `account_level_detection.py`
- - `tweet_level_detection.py`
- - `preprocessing.py`
