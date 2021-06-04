"""
Title: Network Analysis of Replies for the Georgia Election Dataset

Description:
    1. Building a reply network that can be imported into Gephi

"""

# 1. SETUP --------------------------------------------------------------------
import pickle, sys, datetime
import pandas as pd
import numpy as np
import seaborn as sns
m4r_data = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\"
import matplotlib.pyplot as plt
sns.set(font="Arial") # Setting the plot style
sys.path.insert(1, "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_repository")
# Need to install the python-louvain package from taynaud
import community as community_louvain
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
    Building the more complicated Georgia reply network
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
    nodelabels_truncated.columns = ["Node", "Class"]
    # Saving to CSV:
    # nodelabels_truncated.to_csv(m4r_data + "ga_reply_network_truncated_nodelabels.csv", index = False, header = True)
    
    
    
    
    
    
    
    

def centrality():
    df = pd.read_csv(m4r_data + "georgia_reply_network_full.csv")
    G = nx.DiGraph()
    G = nx.from_pandas_edgelist(df, "Source", "Target", ["Weight"], create_using=nx.DiGraph())
    p_central = nx.algorithms.link_analysis.pagerank_alg.pagerank(G)
    in_central = nx.algorithms.centrality.in_degree_centrality(G)
    out_central = nx.algorithms.centrality.out_degree_centrality(G)
    
    d1 = pd.DataFrame().from_dict(p_central, orient = "index", columns = ["PageRank"]).reset_index()
    d2 = pd.DataFrame().from_dict(in_central, orient = "index", columns = ["In-Degree"]).reset_index()
    d3 = pd.DataFrame().from_dict(out_central, orient = "index", columns = ["Out-Degree"]).reset_index()
    centrality_df = ((d1.merge(d2, on = "index")).merge(d3, on = "index")).rename({"index" : "user.id"}, axis = 1)
    
    
    gephi = pd.read_csv(m4r_data + "Georgia Reply Network Louvain Community Detection.csv")
    gephi.columns = ["user.id", "label", "timeset", "Community"]
    gephi = gephi[gephi["Community"].isin([8, 42])]
    
    
    users = pickle.load(open(m4r_data + "us_and_georgia_accounts.p", "rb"))
    
    
    centrality_df = centrality_df.merge(users[["user.id", "predicted_class"]], on = "user.id", how = "left")
    centrality_df = centrality_df.fillna("Unknown")
    centrality_df = centrality_df.merge(gephi[["user.id", "Community"]], how = "left", on = "user.id")
    centrality_df = centrality_df.fillna(0)
    centrality_df.rename({"predicted_class" : "Predicted Class"}, axis = 1, inplace = True)
    centrality_df = centrality_df.merge(users[["user.id", "user.screen_name"]], on = "user.id", how = "left")
    
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4))
    fig.suptitle('Comparing Centrality Measures for the Georgia Reply Network', fontweight = "bold")
    
    sns.scatterplot(ax = axes[0], data = centrality_df, x = "In-Degree", y = "Out-Degree", hue = "Predicted Class", style = "Community", s = 120)
    
    sns.scatterplot(ax = axes[1], data = centrality_df, x = "In-Degree", y = "PageRank", hue = "Predicted Class", style = "Community", s = 120)
    
    handles, labels = axes[0].get_legend_handles_labels()
    
    axes[0].legend([],[], frameon=False)
    axes[1].legend([],[], frameon=False)
    fig.legend(handles, labels, bbox_to_anchor=[1.05, 0.83])
    
    axes[0].set_xlabel("In-Degree Centrality", fontweight = "bold")
    axes[1].set_xlabel("In-Degree Centrality", fontweight = "bold")
    axes[0].set_ylabel("Out-Degree Centrality", fontweight = "bold")
    axes[1].set_ylabel("PageRank Centrality", fontweight = "bold")
    
    plt.subplots_adjust(wspace = 0.25, hspace = 0.1)
    
    plt.savefig(figure_path + "ga_centrality_measures.pdf", bbox_inches = "tight")
    
    plt.show()
    
    #G = nx.DiGraph()
    #G = nx.from_pandas_edgelist(df, "Source", "Target", ["Weight"], create_using=nx.DiGraph())
    # Too Slow b_central = nx.betweenness_centrality(G)

def louvain():
    df = pd.read_csv(m4r_data + "georgia_reply_network_full.csv")
    A = list(df[["Source", "Target"]].to_records(index=False))
    #G = nx.DiGraph()
    G = nx.Graph()
    G.add_edges_from(A)
    for index, row in df.iterrows():
        G.edges[row["Source"], row["Target"]]["weight"] = row["Weight"]
    
    partition = community_louvain.best_partition(G, partition = None, weight = 'weight', resolution = 1.2, random_state = None)
    pdf = pd.DataFrame().from_dict(partition, orient='index', columns = ["Community"])
    pdf["Count"] = 1
    a = pdf.groupby("Community").count().sort_values("Count")
    a["Count"] / sum(a["Count"]) * 100
    
    
    # Compare to Gephi's partitioning using Louvain:
    gephi = pd.read_csv(m4r_data + "Georgia Reply Network Louvain Community Detection.csv")
    
    gephi["Count"] = 1
    c = gephi[[ "modularity_class" , "Count"]].groupby("modularity_class").count().reset_index()
    
    
    plt.figure(figsize=(8, 4), dpi=80)
    plt.scatter(c["modularity_class"], c["Count"]/sum(c["Count"]) );
    plt.title("Sizes of Communities Found in the Georgia Reply Network", fontweight = "bold")
    plt.xlabel("Modularity Class Number", fontweight = "bold")
    plt.ylabel("Proportion of Nodes", fontweight = "bold")
    plt.text(8 + 2, 945 / 3865, "8")
    plt.text(42 + 2, 662 / 3865, "42")
    plt.savefig(figure_path + "louvain_distribution_of_community_sizes.pdf", bbox_inches = "tight")
    plt.show()
    
    # Takes far too long:
    # partition2 = greedy_modularity_communities(G)
    
    
    # G = nx.Graph().from_pandas_dataframe(df, "Source", "Target", ["Weight"])
    
    
    
    # G.add_edges_from(# INSERT LIST OF TUPLES - PAIRS OF NODES)
    # G.edges[1, 2]['weight'] = 4
    
    
    
    democrats = gephi[gephi["modularity_class"] == 8]["Id"]
    republicans = gephi[gephi["modularity_class"] == 42]["Id"]
    df = pickle.load(open(m4r_data + "georgia_election_tweets.p", "rb"))
    dem_tweets = df[df["user.id"].isin(democrats)]
    rep_tweets = df[df["user.id"].isin(republicans)]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey = True)
    fig.suptitle('Comparing Distributions of VADER Polarity Scores for the Two Largest Louvain Groups', fontweight = "bold")
    
    sns.histplot(
        ax = axes[0],
        data = dem_tweets.sort_values("predicted_class"),
        x = "vader",
        hue = "predicted_class",
        stat="density",
        common_norm = False,
        bins = np.linspace(-1,1,20),
        element = "step"
        )
    
    sns.histplot(
        ax = axes[1],
        data = rep_tweets.sort_values("predicted_class"),
        x = "vader",
        hue = "predicted_class",
        stat="density",
        common_norm = False,
        bins = np.linspace(-1,1,20),
        element = "step"
        )
    plt.show()
    
    yi = dem_tweets[(dem_tweets["created_at"] > datetime.datetime(2021, 1, 1, 5, 0)) & (dem_tweets["created_at"] < datetime.datetime(2021, 1, 10,5,0))].index
    plotdata1 = pd.DataFrame()
    plotdata1["vader"] = dem_tweets.loc[yi, "vader"]
    plotdata1["time"] = dem_tweets.loc[yi, "created_at"].dt.round("D")
    plotdata1["predicted_class"] = dem_tweets.loc[yi, "predicted_class"]
    sns.lineplot(data = plotdata1, x = "time", y = "vader", hue = "predicted_class")
    plt.plot()
    
    zi = rep_tweets[(rep_tweets["created_at"] > datetime.datetime(2021, 1, 1, 5, 0)) & (rep_tweets["created_at"] < datetime.datetime(2021, 1, 10,5,0))].index
    plotdata2 = pd.DataFrame()
    plotdata2["vader"] = rep_tweets.loc[zi, "vader"]
    plotdata2["time"] = rep_tweets.loc[zi, "created_at"].dt.round("D")
    plotdata2["predicted_class"] = rep_tweets.loc[zi, "predicted_class"]
    sns.lineplot(data = plotdata2, x = "time", y = "vader", hue = "predicted_class")
    plt.plot()
    plt.show()
    
    
    plotdata1["Community"] = "Group 8"
    plotdata2["Community"] = "Group 42"
    sns.lineplot(data = pd.concat([plotdata1, plotdata2], ignore_index = True), x = "time", y = "vader", hue = "Community")
    plt.show()
    
    
    
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    df = pickle.load(open(m4r_data + "georgia_election_tweets.p", "rb"))
    df = df[df["in_reply_to_status_id"].isna() == False] # keep only retweets
    df = df[df["user.id"].isin(gephi["Id"])]  # keep only users that are in reduced reply network
    users = pickle.load(open(m4r_data + "us_and_georgia_accounts.p", "rb"))
    users = users[users["user.id"].isin(gephi["Id"])]
    edges = df[["user.id", "in_reply_to_user_id", "predicted_class"]].dropna()
    edges.columns = ["Source", "Target", "Source Class"]
    edges["Target"] = edges["Target"].astype("int64")
    edges = edges.merge(users.rename({"user.id" : "Target"}, axis = 1)[["Target", "predicted_class"]], on = "Target", how = "left")
    edges.rename({"predicted_class" : "Target Class"}, axis = 1, inplace = True)
    
    edges = edges.merge(gephi.rename({"Id" : "Source", "modularity_class" : "Source Group"}, axis = 1)[["Source", "Source Group"]], on = "Source", how = "left")
    edges = edges.merge(gephi.rename({"Id" : "Target", "modularity_class" : "Target Group"}, axis = 1)[["Target", "Target Group"]], on = "Target", how = "left")

    edges_backup = edges.copy()

    edges = edges.dropna()
    
    edges = edges[(edges["Target Group"].isin([8, 42])) & (edges["Source Group"].isin([8, 42]))]
    
    CC = edges.groupby(["Source Class", "Target Class", "Source Group", "Target Group"]).count()

def more_complicated_reply_diagram_wth_louvain_groups():
    """
    i.e. the 4 way diagram:
        Human Group 1   |     Human Group 2
        -------------------------------------
        Bot Group 1     |     Bot Group 2
    """









