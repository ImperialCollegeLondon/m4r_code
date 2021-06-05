"""
Title: Network Analysis of Replies for the Georgia Election Dataset

Description:
    1. Building a simple reply network showing interactions across the classes in the Georgia reply network
    2. Building a FULL reply network for the Georgia election dataset that we can calculate centrality scores for
    3. Building a TRUNCATED Georgia reply network from (2.) that can be imported into Gephi to apply Louvain community detection and to visualise
    
"""

# 1. SETUP --------------------------------------------------------------------
import pickle, sys
import pandas as pd
import numpy as np
import seaborn as sns
m4r_data = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\"
import matplotlib.pyplot as plt
sns.set(font="Arial") # Setting the plot style
sys.path.insert(1, "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_repository")
# Need to install the python-louvain package from taynaud
import networkx as nx
figure_path = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\Report\\Figures\\"

def building_simple_reply_network():
    """
    Building simple reply networks for the Georgia and US election datasets
    in order to show interactions between the bot and human classes.
    """
    # Load dataset (change for Georgia or US dataset)
    df = pickle.load(open(m4r_data + "us_election_tweets.p", "rb"))
    # Loading all known users (this is a dataset of the unique users from combining the US and Georgia datasets, and then retrieving the account data of as many of the users that have been replied to in the datasets as possible)
    users = pickle.load(open(m4r_data + "us_and_georgia_accounts.p", "rb"))
    # Dataframe of user ids and class prediction for users that have been replied to; this will be merged with the Georgia dataset.
    users_prediction = (pd.DataFrame({"in_reply_to_user_id" : users["user.id"], "reply_predicted_class" : users["predicted_class"]})).groupby("in_reply_to_user_id").first()
    
    # Producing the simple reply network:
    # Keeping only tweets that are replies from the Georgia dataset
    reply_net = df[["user.id", "in_reply_to_user_id"]].dropna()
    # Merging datasets to retrieve class labels and user names for repliers and recievers of replies
    reply_net = reply_net.merge(users[["user.id", "predicted_class"]], how = "left", on = "user.id")
    reply_net = reply_net.merge(users_prediction, how = "left", on = "in_reply_to_user_id")
    # Dropping unknown users (i.e. unknown if bot or human due to missing account data)
    reply_net = reply_net.dropna()
    # Renaming the columns: i.e. the user replying is the source and the user receiving the reply is the target
    reply_net.columns = ["Source", "Target"]
    reply_net["Count"] = 1 # Adding a count column
    reply_net = reply_net.groupby(["Source", "Target"]).count().reset_index() # Grouping by the source and target and summing up the replies
    
    # Calculating the proportions of bots and humans in the reply network (i.e. how many bots and how many humans are in the reply network?):
    renamer = {"user.id" : "user.id", "in_reply_to_user_id" : "user.id", 'predicted_class' : "class", 'reply_predicted_class' : "class"}
    users_2 = pd.concat( [reply_net[["user.id", "predicted_class"]].rename(renamer, axis = 1) , reply_net[[ "in_reply_to_user_id", "reply_predicted_class"]].rename(renamer, axis = 1) ], ignore_index = True)
    users_2["user.id"] = users_2["user.id"].astype("int64")
    users_2 = users_2.groupby("user.id").first().dropna().reset_index()

def more_complicated_reply_network():
    """
    Building the more complicated Georgia reply network,
    and its truncated counterpart.
    The truncated reply network is built by retrieving the users that recieve more than 9
    replies, and then retrieving all users that interact with the user
    Note that we ignore users replying to themselves.
    """
    # Loading the Georgia tweets
    df = pickle.load(open(m4r_data + "georgia_election_tweets.p", "rb"))
    # Loading all known users from the Georgia and US tweets, and as many of the replied to uers as possible
    users = pickle.load(open(m4r_data + "us_and_georgia_accounts.p", "rb"))
    
    # Creating a dataframe describing the edges
    # The first column is the source (the replier) and the second column is the target (reciever of the reply)
    edges = df[["user.id", "in_reply_to_user_id"]].dropna() # Keeping only tweets that are replies
    edges.columns = ["Source", "Target"] # Renaming columns
    # Removing unknown targets (unknown in_reply_to_user_id's) - this is a rare but possible case; for instance, replies can still exist to tweets that have been deleted
    edges["Target"] = edges["Target"].replace({0 : np.nan}) 
    edges = edges.dropna()
    # Change the Target values to int64 form:
    edges["Target"] = edges["Target"].astype("int64")
    # Setting edge weights to 1
    edges["Weight"] = 1
    # Aggregating the edges (i.e. if there are duplicate replies, we sum the weight up)
    edges = edges.groupby(by = ["Source", "Target"]).Weight.sum().reset_index()
    # Removing people replying to themselves (i.e. threads)
    edges = edges[edges["Source"] != edges["Target"]] # remove people replying to themselves
    # Saving to CSV:
    # edges.to_csv(m4r_data + "ga_reply_network_full.csv", index = False, header = True)
    
    # Creating a dataframe describing the nodes (i.e. assigns a label - their username - to each node)
    nodelabels = pd.DataFrame({"user.id" : list(set(edges.Source).union(set(edges.Target))) })
    nodelabels = nodelabels.merge(users[["user.id", "predicted_class"]], how = "left", on = "user.id")
    
    # BUILDING THE TRUNCATED REPLY NETWORK:
    # Calculating the number of replies each Target recieves:
    replied_to_users = ((edges[["Target", "Weight"]]).groupby("Target").sum().reset_index()).sort_values(by = "Weight")
    # Only keeping interactions with nodes that recieved more than 9 replies:
    edges_truncated = edges[edges["Source"].isin(replied_to_users[replied_to_users["Weight"] > 9].Target)]
    # Saving to CSV:
    #edges_truncated.to_csv(m4r_data + "ga_reply_network_truncated.csv", index = False, header = True)
    # And now the dataframe describing the nodes in the truncated network:
    # i.e. adding a class label (bot or human) (if class label is unknown, fill with 'unknown')
    nodelabels_truncated = ((pd.DataFrame({"user.id" : list(set(edges_truncated.Source).union(set(edges_truncated.Target))) })).merge(users[["user.id", "predicted_class"]], how = "left", on = "user.id")).fillna("unknown")
    nodelabels_truncated.columns = ["Id", "Class"]
    # Saving to CSV:
    # nodelabels_truncated.to_csv(m4r_data + "ga_reply_network_truncated_nodelabels.csv", index = False, header = True)
    

def centrality_scores():
    """
    Calculating and plotting centrality scores for the FULL Georgia Reply Network
    """
    # Retrieving the FULL Georgia Reply Network: (created in more_complicated_reply_network())
    df = pd.read_csv(m4r_data + "ga_reply_network_full.csv")
    
    # Converting this to a networkx graph object:
    G = nx.from_pandas_edgelist(df, "Source", "Target", ["Weight"], create_using=nx.DiGraph())
    
    # Calculating centrality scores:    
    in_central = nx.algorithms.centrality.in_degree_centrality(G) # In-Degree
    out_central = nx.algorithms.centrality.out_degree_centrality(G) # Out-Degree
    p_central = nx.algorithms.link_analysis.pagerank_alg.pagerank(G) # PageRank with alpha = 0.85
    
    # Inserting the scores into a single dataframe:s
    d1 = pd.DataFrame().from_dict(p_central, orient = "index", columns = ["PageRank"]).reset_index()
    d2 = pd.DataFrame().from_dict(in_central, orient = "index", columns = ["In-Degree"]).reset_index()
    d3 = pd.DataFrame().from_dict(out_central, orient = "index", columns = ["Out-Degree"]).reset_index()
    centrality_df = ((d1.merge(d2, on = "index")).merge(d3, on = "index")).rename({"index" : "user.id"}, axis = 1)
    
    # Now retrieving the Louvain community for each account (if the account is in community 8 or 42)
    gephi = pd.read_csv(m4r_data + "Georgia Reply Network Louvain Community Detection.csv")[["Id", "modularity_class"]].rename({"Id" : "user.id", "modularity_class" : "Community"}, axis = 1)
    gephi = gephi[gephi["Community"].isin([8, 42])] # Only care about Community labels for nodes in Communities 8 or 42 (the largest communities)

    # Now retrieving the class (bot or human label)
    users = pickle.load(open(m4r_data + "us_and_georgia_accounts.p", "rb"))[["user.id", "user.screen_name", "predicted_class"]]
    
    # Adding account class and community labels to the centrality score dataframe
    centrality_df = centrality_df.merge(users[["user.id", "predicted_class"]], on = "user.id", how = "left").rename({"predicted_class" : "Class"}, axis = 1)
    centrality_df = centrality_df.fillna("Unknown")
    centrality_df = centrality_df.merge(gephi[["user.id", "Community"]], how = "left", on = "user.id")
    centrality_df = (centrality_df.fillna("Other"))
    centrality_df = centrality_df.merge(users[["user.id", "user.screen_name"]], on = "user.id", how = "left")
    centrality_df["Class"] = centrality_df["Class"].replace({"human" : "Human", "bot" : "Bot"})
    centrality_df["Community"] = centrality_df["Community"].replace({8 : "Group 8", 42 : "Group 42"})
    centrality_df = centrality_df.sort_values(["Class", "Community"], ascending = False)
    
    # Plotting the centrality scores against each other...
    pal = [sns.color_palette("tab10")[7], sns.color_palette("tab10")[0], sns.color_palette("tab10")[1]]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.1), sharey = True)
    fig.suptitle('Comparing Centrality Measures for the Georgia Reply Network', fontweight = "bold")
    # In degree vs Out degree
    sns.scatterplot(ax = axes[0], data = centrality_df, y = "In-Degree", x = "Out-Degree", hue = "Class", style = "Community", s = 120, alpha = 0.9, palette = pal)
    # In degree vs PageRank
    sns.scatterplot(ax = axes[1], data = centrality_df, y = "In-Degree", x = "PageRank", hue = "Class", style = "Community", s = 120, alpha = 0.9, palette = pal)
    
    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    new_handles = [handles[i] for i in [0,2,3,1,4,6,7,5]]
    new_labels = [labels[i] for i in [0,2,3,1,4,6,7,5]]
    axes[0].legend([],[], frameon=False)
    axes[1].legend([],[], frameon=False)
    fig.legend(new_handles, new_labels, loc = "center left",  bbox_to_anchor=[0.67, 0.5])
    
    # Names of axes
    axes[0].set_ylabel("In-Degree Centrality", fontweight = "bold")
    #axes[1].set_ylabel("In-Degree Centrality", fontweight = "bold")
    axes[0].set_xlabel("Out-Degree Centrality", fontweight = "bold")
    axes[1].set_xlabel("PageRank Centrality", fontweight = "bold")
    
    # Adjusting plot size to accommodate legend: right determines how much space is left for the legend - e.g. right = 0.8 leaves 80% of space for legend
    plt.subplots_adjust(right = 0.69, wspace = 0.04, hspace = 0.1)
    
    #plt.savefig(figure_path + "ga_centrality_measures.pdf", bbox_inches = "tight")
    
    plt.show()
    

























