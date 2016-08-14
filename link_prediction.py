#!/usr/bin/env python
# -*- coding: utf-8 -*-


# coding: utf-8

## Link Prediction

# In this notebook, we demonstrate how to use GraphLab Create to construct a simple [Link Prediction]classifier.
# A link prediction classifier is a classifier that can predict the probability of the existence (or non-existence) of a link in a social network (SN). Namely, given a social network, such as Facebook, a link prediction classifier can help to determine if a link between two users exist. Our construction is general and not limited to SN. For example, in a phone call network our classifier can predict whether two subscribers will call each other. 
# 
# There are many diverse methods to construct a link prediction classifier, such as using the SN topology, using the SN users' personal details or using posts and other published online content. 

# This notebook is organized as follows: First, we download a link prediction training dataset and constract a graph from it. Next, we will illustrate how GraphLab's SFrame and SGraph objects can be utilized to extract various SN topological features for each link. Finally, we will demonstrate how to use those features inside a link prediction classifier.

### Downloading the Dataset 

# In this notebook, we will use dataset which was published by [BGU Social Network Research Group](http://proj.ise.bgu.ac.il/sns/datasets.html). 
# We will download the [Google+ dataset](http://proj.ise.bgu.ac.il/sns/googlep.html) which consists of around 3 million links, and generate a SGraph object containing the social links.

import graphlab as gl

# Loading the links dataset into a SFrame object
#sf_links = gl.SFrame.read_csv("https://s3.amazonaws.com/dato-datasets/bgu_directed_network_googleplus/g_plus_pos_and_neg_links.csv.gz")
sf_links = gl.load_sframe('directed_network_googleplus')
# Let's view the data
#print sf_links.head(3)
src = list(sf_links['src'])
dst = list(sf_links['dst'])
edge_all = zip(src, dst)
# Creating SGraph object from the SFrame object
g = gl.SGraph().add_edges(sf_links, src_field='src', dst_field='dst')


# Let use the SGraph's summary function to get details on the loaded SGraph object
#g.summary()


# Our directed graph has around 3 million edges (referred to as links) and over 200,000 vertices (referred to as users). 
# Those links are roughtly divided into two. Positive links (labeled 1), and Negative links (labeled 0). The positive links are observed links in the social network, while negative links were added at random so our binary classifier could learn from the negative examples as well. Note, that if you like to use your own dataset, you will need to add negative (unobserved) edges at random for our construction to work.

### Topological Feature Extraction

# Let's create an SFrame obejct with all user ids.

users_sf = g.get_vertices()
users_sf.rename({"__id": "id"}) 
#users_sf.head(5)


# The users sframe can be used for calculating several topological feature for each link in the dataset. 
# We will start by calculating various simple topological features, such as the each user's in-degree (i.e. the user's number of followers) and user's out-degree (i.e. number of users the user is followling).  
# 
# To calculate each vertex toplogical features, we will first create a SFrame object that contains each user's in-friends (i.e. the users in the network that follows the user), and the user's out-friends (the users in the network which the user follows).

# Calculating each vertices in and out degree
out_friends_sf = sf_links.groupby("src", {"out_friends": gl.aggregate.CONCAT("dst")})
out_friends_sf.rename({"src": "id"})
in_friends_sf = sf_links.groupby("dst", {"in_friends": gl.aggregate.CONCAT("src")})
in_friends_sf.rename({"dst": "id"})


# Using the SFrame [join](http://dato.com/products/create/docs/generated/graphlab.SFrame.join.html#graphlab.SFrame.join) operation, we create a single SFrame which consists of each user's in and out friends.

users_sf = users_sf.join(in_friends_sf, on="id", how="outer")
users_sf = users_sf.join(out_friends_sf, on="id", how="outer")

# we replace missing values with empty lists
users_sf = users_sf.fillna('in_friends',[])
users_sf = users_sf.fillna('out_friends',[])
#users_sf.head(10)


# Using the created SFrame with each user in-friends and out-friends, we calculate several simple topological features for each user,  such as each user's in-degree (i.e. number of followers) and each user's out-degree (i.e. number of users the user is followling). 

def all_friends(in_friends, out_friends):
    in_f = set(in_friends)
    out_f = set(out_friends)
    all_friend = in_f | out_f
    return list(all_friend)
    
''' vertex degree features  '''
''' in_degree_density '''
# Directed graphs
def in_degree_density(in_friends, out_friends):              # Directed graphs
    num_all_friends = len(all_friends(in_friends, out_friends))
    num_in_friends = len(in_friends)
    if  num_all_friends == 0:
        return 0
    return num_in_friends/float(num_all_friends)
    
''' out_degree_density '''
def out_degree_density(in_friends, out_friends):             # Directed graphs
    num_all_friends = len(all_friends(in_friends, out_friends))
    num_out_friends = len(out_friends)
    if  num_all_friends == 0:
        return 0
    return num_out_friends/float(num_all_friends)
    
''' bi_degree_density '''
def bi_degree_density(in_friends, out_friends):           # Directed graphs
    num_all_friends = len(all_friends(in_friends, out_friends))
    num_bi_friends = len(set(in_friends) & set(out_friends))
    if  num_all_friends == 0:
        return 0
    return num_bi_friends/float(num_all_friends)
    
#out_degree - number of users each vertex is following
users_sf['out_degree'] = users_sf["out_friends"].apply(lambda l: len(l) )

#in_degree - number of users following each vertex
users_sf['in_degree'] = users_sf["in_friends"].apply(lambda l: len(l) )

#all_degree - number of uniuqe users that following or are followed by each user
users_sf['all_friends'] = users_sf[['in_friends', 'out_friends']].apply(lambda r: (set(r['in_friends']) | set(r['out_friends'])))
users_sf['all_degree'] = users_sf["all_friends"].apply(lambda l: len(l) )

#bi_degree - number of uniuqe users that are both following and followed by each user
users_sf['bi_friends'] = users_sf[['in_friends', 'out_friends']].apply(lambda r: (set(r['in_friends']) & set(r['out_friends'])))
users_sf['bi_degree'] = users_sf["bi_friends"].apply(lambda l: len(l) )

users_sf['in_degree_density'] = users_sf[['in_friends', 'out_friends']].apply(lambda r: in_degree_density(r['in_friends'],r['out_friends']))
users_sf['out_degree_density'] = users_sf[['in_friends', 'out_friends']].apply(lambda r: out_degree_density(r['in_friends'],r['out_friends']))
users_sf['bi_degree_density'] = users_sf[['in_friends', 'out_friends']].apply(lambda r: bi_degree_density(r['in_friends'],r['out_friends']))

#users_sf.head(10)


# Now, we  have several degree feautres for each user. Lets utilize these users' features and create features for each link in our positive and negative links dataset.
# Namely, for each link in the data conists of two users <i>u</i> and <i>v</i>, we will create SFrame with each user's degree features.


sf_links = sf_links.join(users_sf, on={"src": "id"}, how="right")
sf_links.rename({"in_friends": "src_in_friends", "out_friends": "src_out_friends",'in_degree_density':'src_in_degree_density',
           "all_friends": "src_all_friends", "all_degree": "src_all_degree",'out_degree_density':'src_out_degree_density',
           "bi_friends": "src_bi_friends", "bi_degree": "src_bi_degree",'bi_degree_density':'src_bi_degree_density',
           "in_degree": "src_in_degree", "out_degree": "src_out_degree"
           })

sf_links = sf_links.join(users_sf, on={"dst": "id"}, how="right")
sf_links.rename({"in_friends": "dst_in_friends", "out_friends": "dst_out_friends",'bi_degree_density':'dst_bi_degree_density',
           "all_friends": "dst_all_friends", "all_degree": "dst_all_degree",'in_degree_density':'dst_in_degree_density',
           "bi_friends": "dst_bi_friends", "bi_degree": "dst_bi_degree",'out_degree_density':'dst_out_degree_density',
           "in_degree": "dst_in_degree", "out_degree": "dst_out_degree"})

#sf_links.head(10)


# Beside adding each link's users <i> u </i> and <i> v </i> degree features, to create a decent link prediction classifier, we also need to add features based on the strength of connection between the users.
# In this notebook, we will add for each link three simple type of features:
# - <i>Common-Friends Features</i> - the number of friends both <i>u</i> and <i>v</i> have in common. 
# - <i>Total-Friends Features</i> - the number of friends both <i>u</i> and <i>v</i> have in together.
# - <i>Jaccard coefficient</i>- the number of Common-Friends divided by the Number of Total-Friends.
# 
# Lets define the each feature type function:

def common_friends(u, v, u_friends, v_friends):
    u_friends = set(u_friends)
    if v in u_friends:
            u_friends.remove(v)

    v_friends = set(v_friends)
    if u in v_friends:
        v_friends.remove(u)
    return len(u_friends & v_friends)

def total_friends(u, v, u_friends, v_friends):
    u_friends = set(u_friends)
    if v in u_friends:
        u_friends.remove(v)

    v_friends = set(v_friends)
    if u in v_friends:
        v_friends.remove(u)

    return len(u_friends | v_friends)

def jacc_coef(u,v, u_friends, v_friends):
    t = total_friends(u,v,u_friends,v_friends)
    if  t == 0:
        return 0
    return common_friends(u,v,u_friends, v_friends)/ float(t)

'''Friends-measure'''
def friends_measure(u,v,u_friends,v_friends,G):
     if (u_friends == None)or (v_friends == None):
        return 0
     else:
        u_friends = set(u_friends)
        if v in u_friends:
                u_friends.remove(v)
        v_friends = set(v_friends)
        if u in v_friends:
            v_friends.remove(u)
#        edge = G.get_edges()
#        sub_src = list(edge['__src_id'])
#        sub_dst = list(edge['__dst_id'])
#        edge_all = zip(sub_src, sub_dst)
        deta = 0
        for x in u_friends:
            for y in v_friends:
                if (x == y) or ((x,y) or (y,x) in edge_all):
                    deta += 1
                else:
                    deta = 0
        return deta
# Using these three features type we created 12 new features (4 feature for each feature type) that are based on direction of the friendship between 
# <i>u</i> and <i>v</i>and their friends. 
# 
# Please note that The formal mathematical defintion of the feature presented throught this section can be found in [Fire et al. 2014](http://dl.acm.org/citation.cfm?id=2542192).


sf_links['common_friends'] = sf_links[['src','dst', 'src_all_friends', 'dst_all_friends']].apply(lambda r: common_friends(r['src'], r['dst'],r['src_all_friends'], r['dst_all_friends']))
sf_links['common_bi_friends'] = sf_links[['src','dst', 'src_bi_friends', 'dst_bi_friends']].apply(lambda r: common_friends(r['src'], r['dst'],r['src_bi_friends'], r['dst_bi_friends']))
sf_links['common_in_friends'] = sf_links[['src','dst', 'src_in_friends', 'dst_in_friends']].apply(lambda r: common_friends(r['src'], r['dst'],r['src_in_friends'], r['dst_in_friends']))
sf_links['common_out_friends'] = sf_links[['src','dst', 'src_out_friends', 'dst_out_friends']].apply(lambda r: common_friends(r['src'], r['dst'],r['src_out_friends'], r['dst_out_friends']))
#
sf_links['total_friends'] = sf_links[['src','dst', 'src_all_friends', 'dst_all_friends']].apply(lambda r: total_friends(r['src'], r['dst'],r['src_all_friends'], r['dst_all_friends']))
sf_links['total_bi_friends'] = sf_links[['src','dst', 'src_bi_friends', 'dst_bi_friends']].apply(lambda r: total_friends(r['src'], r['dst'],r['src_bi_friends'], r['dst_bi_friends']))
sf_links['total_in_friends'] = sf_links[['src','dst', 'src_in_friends', 'dst_in_friends']].apply(lambda r: total_friends(r['src'], r['dst'],r['src_in_friends'], r['dst_in_friends']))
sf_links['total_out_friends'] = sf_links[['src','dst', 'src_out_friends', 'dst_out_friends']].apply(lambda r: total_friends(r['src'], r['dst'],r['src_out_friends'], r['dst_out_friends']))


sf_links['jacc_coef'] = sf_links[['src','dst', 'src_all_friends', 'dst_all_friends']].apply(lambda r: jacc_coef(r['src'], r['dst'],r['src_all_friends'], r['dst_all_friends']))
sf_links['bi_jacc_coef'] = sf_links[['src','dst', 'src_bi_friends', 'dst_bi_friends']].apply(lambda r: jacc_coef(r['src'], r['dst'],r['src_bi_friends'], r['dst_bi_friends']))
sf_links['in_jacc_coef'] = sf_links[['src','dst', 'src_in_friends', 'dst_in_friends']].apply(lambda r: jacc_coef(r['src'], r['dst'],r['src_in_friends'], r['dst_in_friends']))
sf_links['out_jacc_coef'] = sf_links[['src','dst', 'src_out_friends', 'dst_out_friends']].apply(lambda r: jacc_coef(r['src'], r['dst'],r['src_out_friends'], r['dst_out_friends']))

sf_links['friends_measure'] = sf_links[['src','dst', 'src_all_friends', 'dst_all_friends']].apply(lambda r: friends_measure(r['src'], r['dst'],r['src_all_friends'], r['dst_all_friends'],g))
sf_links['bi_friends_measure'] = sf_links[['src','dst', 'src_bi_friends', 'dst_bi_friends']].apply(lambda r: friends_measure(r['src'], r['dst'],r['src_bi_friends'], r['dst_bi_friends'],g))
sf_links['in_friends_measure'] = sf_links[['src','dst', 'src_in_friends', 'dst_in_friends']].apply(lambda r: friends_measure(r['src'], r['dst'],r['src_in_friends'], r['dst_in_friends'],g))
sf_links['out_friends_measure'] = sf_links[['src','dst', 'src_out_friends', 'dst_out_friends']].apply(lambda r: friends_measure(r['src'], r['dst'],r['src_out_friends'], r['dst_out_friends'],g))

#sf_links.head(10)


# Similar to the above method, we can also extract for each link, such as each user [PageRank](https://dato.com/products/create/docs/generated/graphlab.pagerank.create.html) and [neighborhood subgraph](https://dato.com/products/create/docs/generated/graphlab.SGraph.get_neighborhood.html?highlight=neighborhood#graphlab.SGraph.get_neighborhood) size. We let the reader to try to add these features (and  maybe some additional features) by themselves. 
# 
# Let us move to the next section, in which we explain how the extracted links' features can be utilized to create a link prediction classifier.

### Constructing a Link Prediction Classifier

# In order to create a link prediction classifier using our constructed links dataset (sf), lets first randomly split our dataset into training that contains 20% of the dataset, and testing datasets that contains 80% of the dataset.


train, test = sf_links.random_split(0.2)


# We now can use GraphLab Create's [classfication toolkit](https://dato.com/products/create/docs/graphlab.toolkits.classifier.html#creating-a-classifier) and create and evaluate a link prediction classifier based only on <i>u</i> and <i>v</i> degree features:
# 

degree_features_list = [c for c in train.column_names() if "degree" in c]
#print "Degree Features %s" % degree_features_list
#cls = gl.classifier.create(train, target="class",max_depth = 5)
#results = cls.evaluate(test)
#print results


# We get pretty good accuracy of 0.88. Let us add also the link based features and use the Boosted-Trees classifier to create and evaluate a link predicdiction classifier.


link_features_list = ['common_friends', 'common_in_friends', 'common_out_friends', 'common_bi_friends', 'total_friends', 'total_in_friends', 'total_out_friends', 'total_bi_friends',
'jacc_coef', 'bi_jacc_coef', 'in_jacc_coef', 'out_jacc_coef']
all_features_list = degree_features_list +   link_features_list
cls = gl.boosted_trees_classifier.create(train,target="class",features=all_features_list)
results = cls.evaluate(test)
print results


# Using the additional link features, we got considerbly better accuracy of around 0.946. We can try to further improve the accuracy by adding additional features or by increasing the size of the training dataset.
