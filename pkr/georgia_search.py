from twarc import Twarc2, expansions
import datetime
import json
import pytz
import time

get_counts = False
do_search = True
file_name = 'tweets11923.txt'
# Specify the start time in UTC for the time period you want replies from
start_time = datetime.datetime(2021, 1, 1, 9, 0, 0, 0)
# Specify the end time in UTC for the time period you want Tweets from
end_time =   datetime.datetime(2021, 1, 1, 23, 59, 59, 59)

# Replace your bearer token below
client = Twarc2(bearer_token="AAAAAAAAAAAAAAAAAAAAACfFSAEAAAAAxZz8O%2Bw7byEiw30MQrg3vB3fgdQ%3D9BTI4iWdjeBm7ESNocjfiZ6yLWltO2ZBnpn9TFs1drcWHfgLVi")
client.metadata = False

eastern = pytz.timezone('US/Eastern')
start_time = eastern.localize(start_time)
end_time = eastern.localize(end_time)

# This is where we specify our query as discussed in module 5
query = "from:twitterdev"

# Name and path of the file where you want the Tweets written to
search_terms = "#georgia OR #gapol OR #ossoff OR #warnock OR #perdue OR #loeffler OR #georgiarunoffs OR #gasen OR #gasenateraces OR @ossoff OR @sendavidperdue OR @kloeffler OR @reverendwarnock"
# The search_all method call the full-archive search endpoint to get Tweets based on the query, start and end times


fields = ["author_id", "created_at", "entities", "geo", "id", "in_reply_to_user_id", "lang", "public_metrics", "referenced_tweets", "source", "text"]
if do_search:
    search_results = client.search_all(query=search_terms, start_time=start_time, end_time=end_time, max_results=100)

    # Twarc returns all Tweets for the criteria set above, so we page through the results
    count = 0
    counts = []
    for page in search_results:
        count+=1
        counts.append(len(page))
        print("count=",count)
        # The Twitter API v2 returns the Tweet information and the user, media etc.  separately
        # so we use expansions.flatten to get all the information in a single JSON
        result = expansions.flatten(page)
        # We will open the file and append one JSON object per new line
        with open(file_name, 'a+') as filehandle:
            for tweet in result:
                if 'context_annotations' in tweet: del tweet['context_annotations']
                if 'entities' in tweet:
                    if 'annotations' in tweet['entities']: del tweet['entities']['annotations']
                filehandle.write('%s\n' % json.dumps(tweet))
        if count==290:
            print("sleeping..., count=",count,)
            for i in range(15):
                print("minute ",i)
                time.sleep(60)
            print("...finished sleeping")
            count=0

if get_counts:
    file_name2='countsD.txt'

    count_results = client.counts_all(query=search_terms, start_time=start_time, end_time=end_time,granularity='day')
    count = 0
    counts = []
    for page in count_results:
        count+=1
        counts.append(len(page))
        print("count=",count)
        # The Twitter API v2 returns the Tweet information and the user, media etc.  separately
        # so we use expansions.flatten to get all the information in a single JSON
        data = expansions.flatten(page)
        # We will open the file and append one JSON object per new line
        with open(file_name2, 'a+') as filehandle:
            for item in data:
                filehandle.write('%s\n' % json.dumps(item))
