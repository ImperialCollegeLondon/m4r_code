"""
Tweet Tokenizer

Rules/Hierarchy:
    urls & media -> "<url>"
    user mentions -> "<user>"
    emojis -> <emoji> or <smile> or <heart> or <lolface> or <neutralface> or <angryface>
    hashtags -> "<hashtag>"
    number -> "<number>"
    repetition -> "<repeat>"
    all caps -> "<allcaps>"
"""

from nltk import TweetTokenizer
import emoji
from itertools import groupby

# - FUNCTIONS TO CHECK FOR THE VARIOUS RULES -

# 1. Replaces all URLs with <urls>
def url_replace(tweet, urls):
    """
    Parameters: tweet and entities data ("urls" and "media")
    Output: tweet with all urls and media replaced with "<url>"
    Note that all urls have a length strictly greater than 5
    """
    for [ut, beg, end] in urls:
        replacement = "<url>" + (" "*(len(ut)-5))
        tweet = tweet[:beg] + replacement + tweet[end:]
    return tweet


# Dictionaries of various emoji categories:
# <smile>
smile_set = {"😀","😃","😄","😁","🙂","😉","😊","😇","🥰","😍","🤩","☺",
             "😋","😛","🤗","😌","🥳","😎"}

# <heart>
heart_set = {"💘","💝","💖","💗","💓","💞","💕","💟","❣","💔","❤","🧡",
             "💛","💚","💙","💜","🤎","🖤","🤍"}

# <lolface>
lol_set = {"😆","😅","🤣","😂", "😝", "🤪", "😜", "🤭"}

# <neutralface>
neutral_set = {"😐", "😑", "😶", "🤨", "🤐", "🤔", "😬", "😕", "🙃"}

# <angryface>
angry_set= {"😤","😡","😠","🤬","👿"}



# - TOKENIZER -
def tweet_tokeniser(tweet, url_data, user_mentions_data):
    """
    Parameters:
        tweet (full text from the status or retweeted status)
        url_data (list of url data from entities, in form [[url, beg, end], ...])
        user_mention_data (list of user_mention handles, in form [handle, handle, ...])
    Output:
        list of tokens in form [token, token, ...]
    """
    # 1. Replacing urls:
    tweet = url_replace(tweet, url_data)
    
    # 2. Tokenising:
    tknzr = TweetTokenizer(preserve_case=True, strip_handles=False)
    tokens = tknzr.tokenize(tweet)
    
    new_tokens = []
    
    num_urls = 0
    num_hashtags = 0
    num_mentions = 0
    num_emojis = 0
    
    for tok in tokens:
        # URL?
        if tok == "<url>":
            new_tokens.append("<url>")
            num_urls += 1
        # User Mention?
        elif tok[0] == "@" and tok[1:] in user_mentions_data:
            new_tokens.append("<user>")
            num_mentions += 1
        # Emoji?
        elif tok in emoji.UNICODE_EMOJI:
            if tok in smile_set:
                new_tokens.append("<smile>")
            elif tok in heart_set:
                new_tokens.append("<heart>")
            elif tok in angry_set:
                new_tokens.append("<angryface>")
            elif tok in lol_set:
                new_tokens.append("<lolface>")
            elif tok in neutral_set:
                new_tokens.append("<neutralface>")
            else:
                new_tokens.append(tok)
                #new_tokens.append("<emoji>")
            num_emojis += 1
        # Hashtag?
        elif tok[0] == "#" and len(tok)>1:
            if tok[1] != "#":
                new_tokens.append("<hashtag>")
                new_tokens.append(tok[1:])
                num_hashtags += 1
            else:
                new_tokens.append(tok)
        else:
            # Number?
            try:
                float(tok)
                new_tokens.append("<number>")
            except:
                temp_tok = tok.lower()
                new_tok = ''.join( "".join(g)[:2] for k, g in groupby(temp_tok))
                if tok.isupper() and len(tok)>1:
                    if new_tok == temp_tok:
                        new_tokens.append(temp_tok)
                        new_tokens.append("<allcaps>")
                    else:
                        new_tokens.append(new_tok)
                        new_tokens.append("<allcaps>")
                        new_tokens.append("<repeat>")
                else:
                    if new_tok == temp_tok:
                        new_tokens.append(tok)
                    else:
                        new_tokens.append(new_tok)
                        new_tokens.append("<repeat>")
                        
    return new_tokens, num_hashtags, num_mentions, num_urls, num_emojis
                    
                
                

