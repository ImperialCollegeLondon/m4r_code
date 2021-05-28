"""
Second Tweet Tokeniser

Takes the full text without any metadata and tokenises it based solely on textual features
"""


# Importing packages
from nltk import TweetTokenizer
import emoji
from itertools import groupby
import re

# This is the file path for the pretrained GloVe embeddings 
path_to_m4r = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\"
path_to_embeddings = path_to_m4r + "m4r_data\\glove.twitter.27B\\glove.twitter.27B.25d.txt"

# Smiley Faces for regex (regular expression)
# From stanford nlp glove ruby code (https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb)
eyes = "[8:=;]"
nose = "['`\-]?"

# Dictionaries of various emoji categories:
# <smile>
smile_set = {"😀","😃","😄","😁","🙂","😉","😊","😇","🥰","😍","🤩","☺",
             "😋","😛","🤗","😌","🥳","😎", ":)", "(:", ":-)", "(-:", "8)", "^.^", "^_^", ":]", "[:"}

# <heart>
heart_set = {"💘","💝","💖","💗","💓","💞","💕","💟","❣","💔","❤","🧡",
             "💛","💚","💙","💜","🤎","🖤","🤍", "<3"}

# <lolface>
lol_set = {"😆","😅","🤣","😂", "😝", "🤪", "😜", "🤭", ":D", ":-D", ":P", ":p", ";)", ";-)"}

# <neutralface>
neutral_set = {"😐", "😑", "😶", "🤨", "🤐", "🤔", "😬", "😕", "🙃", ":|", "-_-", ":-|"}

# <angryface> # apparently not in glove? but sadface is
angry_set= {"😤","😡","😠","🤬","👿", "😞", "😥", "😟", "😓", "😢", "😭", "🙁", "☹️", "😩", "😰", "😨", "😠", ":(", "):", ">:(", "):<", ":'(", ")':", ">_<", ":'-(", ")-':", ":-(", ")-:", ":/", ":-/", "/:"}

# <sadface>

#reactions_set = set(list(smile_set) + list(heart_set) + list(lol_set) + list(neutral_set) + list(angry_set))

# Creating the set of emojis that appear in the GloVe embeddings:
glove_emojis = set()
with open(path_to_embeddings, 'r', encoding = "utf-8") as t:
    for line in t:
        values = line.split()
        word = values[0]
        if word in emoji.UNICODE_EMOJI:
            glove_emojis.add(word)
            

def check_url(token):
    """
    Checks if a token is a URL
    (seen in https://www.w3resource.com/python-exercises/re/python-re-exercise-42.php)
                     'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    """
    url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', token)
    
    if len(url) == 0:
        return False
    else:
        return True
    

def check_number(token):
    """
    Checks if a token is a number
    """
    try:
        float(token)
        return True
    except:
        return False


def check_mention(token):
    """
    Checks if a token is a user mention
    """
    # Removes user handle
    temp = TweetTokenizer(strip_handles=True)
    result = temp.tokenize(token)
    if result == []:
        return True
    else:
        return False


emoticons = {":(", "):", ">:(", "):<", ":'(", ")':", ">_<", ":|", "-_-",  ":D", ":-D", ":)", "(:", ":-)", "(-:", "<3", ">_<", ":'-(", ")-':", ":-(", ")-:", ":/", ":-/", "/:", ":-|", ":P", ":p", ";)", ";-)", "8)", "^.^", "^_^", ":]", "[:"}

def check_emoji(token):
    """
    Checks if a token is an emoji
    """
    if token in emoji.UNICODE_EMOJI:
        return True
    elif token in emoticons:
        return True
    else:
        return False
    

def check_hashtag(token):
    """
    Checks if a token is a hashtag
    """
    if len(token)>1:
        if token[0] == "#" and token[1] != "#":
            return True
        else:
            return False
    else:
        return False
    

def check_allcaps(token):
    """
    Checks if a token is in all caps
    """
    if token.isupper():
        return True
    else:
        return False
    

def replace_repeats(token):
    """
    Checks if a token contains repeated characters
    ! tweet tokenizer removes repeated punctuation and just returns a list of up to
    three single character strings
    i.e. !!!!!!!!! -> "!", "!", "!"
    """
    # changing to lower case:
    token = token.lower()
    # grouping together successive characters:
    counting = [[k, len(list(g))] for k,g in groupby(token)]
    # piecing the new token together:
    new_token = ""
    for [x,y] in counting:
        # glove twitter vocabulary doesn't contain "..." but does contain "…"
        if x == "...":
            return "…"
        # check if character contains 2 or more successive repeats
        elif y >=  2:
            new_token += x*2
        # else character only appears once in the immediate area
        else:
            new_token += x*y
    
    # ADD CODE IN HERE TO GO THROUGH ALL OF THE POSSIBILITIES AVAILABLE
    # AND SEE IF ANY APPEAR IN THE GLOVE VOCABULARY
    # E.G. if new_token in glove_vocabulary:
    #           return new_token
    #      else:
    #        cycle through all of the possibilities and check if in glove_vocab?
    
    return new_token
                    

def check_emoticon(token):
    
    return False


def replace_emoji(tok):
    if tok in smile_set:
        return "<smile>"
    elif tok in heart_set:
        return "<heart>"
    elif tok in angry_set:
        return "<sadface>" # "<angryface>"
    elif tok in lol_set:
        return "<lolface>"
    elif tok in neutral_set:
        return "<neutralface>"
    else:
        return "<emoji>" # not an actual token, but will be useful to count number of emojis...
    




def text_tokeniser(text, preserve_case = False):
    # Creating the TweetTokenizer object:
    tknzr = TweetTokenizer(preserve_case=True, strip_handles=False)
    # Tokenising the tweet:
    token_list = tknzr.tokenize(text)
    # Creating empty token list to store the refined tokens:
    new_token_list = []
    
    
    for token in token_list:
        # Check URL:
        if check_url(token):
            new_token_list.append("<url>")
        elif check_mention(token):
            new_token_list.append("<user>")
        elif check_emoji(token):
            if token in glove_emojis:
                new_token_list.append(token)
            else:
                new_token_list.append( replace_emoji(token) )
        elif check_emoticon(token):
            new_token_list.append("emoticon") # change!
        elif check_hashtag(token):
            new_token_list.append("<hashtag>")
            new_token_list.append(token[1:].lower())
        elif check_number(token):
            new_token_list.append("<number>")
        else:
            if check_allcaps(token):
                new_token_list.append("<allcaps>")
            temp_token = token.lower()
            token = replace_repeats(token)
            if token == temp_token:
                new_token_list.append(token)
            else:
                new_token_list.append("<repeat>")
                new_token_list.append(token)
                
    return " ".join(new_token_list)
                
            
        
    
# Additionally we need to output metadata (i.e. tally of how many hashtags, user mentions, urls used etc.) 
    
    
    
    
    
    
    # Substituting URLS for <url> (Twitter corpus is all lower case!)
    
    

