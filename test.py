# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 10:31:14 2016

@author: hh
"""

import graphlab as gl

#sf_links = gl.SFrame.read_csv("https://s3.amazonaws.com/dato-datasets/bgu_directed_network_googleplus/g_plus_pos_and_neg_links.csv.gz")
sf_links = gl.load_sframe('directed_network_googleplus')[:500]
g = gl.SGraph().add_edges(sf_links, src_field='src', dst_field='dst')
users_sf = g.get_vertices()
users_sf.rename({"__id": "id"}) 

out_friends_sf = sf_links.groupby("src", {"out_friends": gl.aggregate.CONCAT("dst")})
out_friends_sf.rename({"src": "id"})
in_friends_sf = sf_links.groupby("dst", {"in_friends": gl.aggregate.CONCAT("src")})
in_friends_sf.rename({"dst": "id"})

users_sf = users_sf.join(in_friends_sf, on="id", how="outer")
users_sf = users_sf.join(out_friends_sf, on="id", how="outer")

# we replace missing values with empty lists
users_sf = users_sf.fillna('in_friends',[])
users_sf = users_sf.fillna('out_friends',[])

######################################################################
''' vertex features  '''

''' all_friends '''
def all_friends(in_friends, out_friends):
    in_f = set(in_friends)
    out_f = set(out_friends)
    all_friend = in_f | out_f
    return all_friend


''' bi_friends '''
def bi_friends(in_friends, out_friends):
    in_f = set(in_friends)
    out_f = set(out_friends)
    bi_friend = in_f & out_f
    return bi_friend
    
'''all_friends+'''
def all_friends_plus(ids, in_friends, out_friends):
    aa = {ids}
    allf = set(all_friends(in_friends, out_friends))
    all_friend_plus = allf | aa
    return all_friend_plus

'''nh-subgraph'''
def nh_subgraph(idl, G):
#    nh_sub_list = []
    subg = G.get_neighborhood(ids = idl, radius = 2, full_subgraph = True)
    sub_edge = subg.get_edges()
    sub_src = list(sub_edge['__src_id'])
    sub_dst = list(sub_edge['__dst_id'])
    nh_sub = zip(sub_src, sub_dst)
    for i in nh_sub:
        if idl in i:
            nh_sub.remove(i)
#    for j in nh_sub:
#        nh_sub_list.extend(j)
    return nh_sub
    
'''nh-subgraph+'''    
def nh_subgraph_plus(idl, G):
#    nh_sub_list = []
    subg = G.get_neighborhood(ids = idl, radius = 2, full_subgraph = True)
    sub_edge = subg.get_edges()
    sub_src = list(sub_edge['__src_id'])
    sub_dst = list(sub_edge['__dst_id'])
    nh_sub = zip(sub_src, sub_dst)
#    for j in nh_sub:
#        nh_sub_list.extend(j)
    return nh_sub

####################################################################   
    
users_sf['all_friends'] = users_sf[['in_friends', 'out_friends']].apply(lambda r: all_friends(r['in_friends'],r['out_friends']))
users_sf['bi_friends'] = users_sf[['in_friends', 'out_friends']].apply(lambda r: bi_friends(r['in_friends'],r['out_friends']))
users_sf['all_friends_plus'] = users_sf[['id','in_friends', 'out_friends']].apply(lambda r: all_friends_plus(r['id'],r['in_friends'],r['out_friends']))
users_sf['nh_subgraph'] = users_sf[['id']].apply(lambda r: nh_subgraph(r['id'],g))
users_sf['nh_subgraph_plus'] = users_sf[['id']].apply(lambda r: nh_subgraph_plus(r['id'],g))

####################################################################    
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
    
#################################################################### 
users_sf['in_degree_density'] = users_sf[['in_friends', 'out_friends']].apply(lambda r: in_degree_density(r['in_friends'],r['out_friends']))
users_sf['out_degree_density'] = users_sf[['in_friends', 'out_friends']].apply(lambda r: out_degree_density(r['in_friends'],r['out_friends']))
users_sf['bi_degree_density'] = users_sf[['in_friends', 'out_friends']].apply(lambda r: bi_degree_density(r['in_friends'],r['out_friends']))

#################################################################### 
''' vertex subgraphs features  '''
''' density-nh-subgraph '''
def nh_subgraph_density(idl, G, in_friends, out_friends):
    num_all_friends = len(all_friends(in_friends, out_friends))
    num_nh_subgraph = len(nh_subgraph(idl, G))
    if  num_nh_subgraph == 0:
        return 0
    return float(num_all_friends)/num_nh_subgraph
    
''' density-nh-subgraph+ '''
def nh_subgraph_density_plus(idl, G, in_friends, out_friends):
    num_all_friends = len(all_friends(in_friends, out_friends))
    num_nh_subgraph_plus = len(nh_subgraph_plus(idl, G))
    if  num_nh_subgraph_plus == 0:
        return 0
    return float(num_all_friends)/num_nh_subgraph_plus
    
''' scc-nh-subgraph'''
def avg_scc(idl, G, in_friends, out_friends):          # Directed graphs
    num_all_friends = len(all_friends(in_friends, out_friends))
    nh_subg = nh_subgraph(idl, G)
    scc_num = []
    for u,v in nh_subg:
        if (u,v) and (v,u) in nh_subg:
            scc_num.append((u,v))
    scc_num_sub = float(len(scc_num))/2
    if scc_num_sub == 0:
        return 0
    return float(num_all_friends)/scc_num_sub

''' wcc-nh-subgraph'''
def avg_wcc(idl, G, in_friends, out_friends):             # Directed graphs
    num_all_friends = len(all_friends(in_friends, out_friends))
    nh_subg = nh_subgraph(idl, G)
    wcc_num = []
    for u,v in nh_subg:
        if (u,v) and (v,u) not in nh_subg:
            wcc_num.append((u,v))
    wcc_num_sub = len(wcc_num)
    if wcc_num_sub == 0:
        return 0
    return float(num_all_friends)/wcc_num_sub
    
''' scc-nh-subgraph'''
def avg_scc_plus(idl, G, in_friends, out_friends):            # Directed graphs
    num_all_friends = len(all_friends(in_friends, out_friends))
    nh_subg = nh_subgraph_plus(idl, G)
    scc_num = []
    for u,v in nh_subg:
        if (u,v) and (v,u) in nh_subg:
            scc_num.append((u,v))
    scc_num_sub_plus = float(len(scc_num))/2
    if scc_num_sub_plus == 0:
        return 0
    return float(num_all_friends)/scc_num_sub_plus
    
#################################################################### 
users_sf['nh_subgraph_density'] = users_sf[['id','in_friends', 'out_friends']].apply(lambda r: nh_subgraph_density(r['id'], g ,r['in_friends'],r['out_friends']))
users_sf['nh_subgraph_density_plus'] = users_sf[['id','in_friends', 'out_friends']].apply(lambda r: nh_subgraph_density_plus(r['id'], g ,r['in_friends'],r['out_friends']))
users_sf['avg_scc'] = users_sf[['id','in_friends', 'out_friends']].apply(lambda r: avg_scc(r['id'], g ,r['in_friends'],r['out_friends']))
users_sf['avg_wcc'] = users_sf[['id','in_friends', 'out_friends']].apply(lambda r: avg_wcc(r['id'], g ,r['in_friends'],r['out_friends']))
users_sf['avg_scc_plus'] = users_sf[['id','in_friends', 'out_friends']].apply(lambda r: avg_scc_plus(r['id'], g ,r['in_friends'],r['out_friends']))
#################################################################### 

sf_links = sf_links.join(users_sf, on={"src": "id"}, how="right")
sf_links.rename({"in_friends": "src_in_friends", "out_friends": "src_out_friends","bi_friends": "src_bi_friends",
           "all_friends": "src_all_friends", "all_friends_plus": "src_all_friends_plus",
           "nh_subgraph": "src_nh_subgraph", "nh_subgraph_plus": "src_nh_subgraph_plus",
           "in_degree_density": "src_in_degree_density", "out_degree_density": "src_out_degree_density",
           "bi_degree_density": "src_bi_degree_density", "nh_subgraph_density": "src_nh_subgraph_density",
           "nh_subgraph_density_plus": "src_nh_subgraph_density_plus", "avg_scc": "src_avg_scc",
           "avg_wcc": "src_avg_wcc", "avg_scc_plus": "src_avg_scc_plus"
           })

sf_links = sf_links.join(users_sf, on={"dst": "id"}, how="right")
sf_links.rename({"in_friends": "dst_in_friends", "out_friends": "dst_out_friends","bi_friends": "dst_bi_friends",
           "all_friends": "dst_all_friends", "all_friends_plus": "dst_all_friends_plus",
           "nh_subgraph": "dst_nh_subgraph", "nh_subgraph_plus": "dst_nh_subgraph_plus",
           "in_degree_density": "dst_in_degree_density", "out_degree_density": "dst_out_degree_density",
           "bi_degree_density": "dst_bi_degree_density", "nh_subgraph_density": "dst_nh_subgraph_density",
           "nh_subgraph_density_plus": "dst_nh_subgraph_density_plus", "avg_scc": "dst_avg_scc",
           "avg_wcc": "dst_avg_wcc", "avg_scc_plus": "dst_avg_scc_plus"
           })

#################################################################### 
''' edge features'''    
'''common-friends'''

def common_friends(u, v, u_friends, v_friends):
    u_friends = set(u_friends)
    if v in u_friends:
            u_friends.remove(v)

    v_friends = set(v_friends)
    if u in v_friends:
        v_friends.remove(u)
    return len(u_friends & v_friends)

'''total-friends'''
def total_friends(u, v, u_friends, v_friends):
    u_friends = set(u_friends)
    if v in u_friends:
        u_friends.remove(v)

    v_friends = set(v_friends)
    if u in v_friends:
        v_friends.remove(u)
    return len(u_friends | v_friends)

'''Jaccard's coefficient'''
def jacc_coef(u,v, u_friends, v_friends):
    t = total_friends(u,v,u_friends,v_friends)
    if  t == 0:
        return 0
    return common_friends(u,v,u_friends, v_friends)/ float(t)
    
'''transitive friends'''
def transitive_friends(u, v, u_friends, v_friends):        # Directed graphs
    u_friends = set(u_friends)
    v_friends = set(v_friends)
    return len(u_friends & v_friends)
    
'''preferential-attachment-score'''

def attachment_score(u, v, u_friends, v_friends):
    u_friends = set(u_friends)
    if v in u_friends:
            u_friends.remove(v)

    v_friends = set(v_friends)
    if u in v_friends:
        v_friends.remove(u)
    return len(u_friends)*len(v_friends)    
    
'''Friends-measure'''
def friends_measure(u,v,u_friends,v_friends,G):
    u_friends = set(u_friends)
    if v in u_friends:
            u_friends.remove(v)
    v_friends = set(v_friends)
    if u in v_friends:
        v_friends.remove(u)
    edge = G.get_edges()
    sub_src = list(edge['__src_id'])
    sub_dst = list(edge['__dst_id'])
    edge_all = zip(sub_src, sub_dst)
    deta = 0
    for x in u_friends:
        for y in v_friends:
            if (x == y) or ((x,y) or (y,x) in edge_all):
                deta += 1
            else:
                deta = 0
    return deta

'''opposite direction friends'''
def opp_dire_friends(u,v,G):
    edge = G.get_edges()
    sub_src = list(edge['__src_id'])
    sub_dst = list(edge['__dst_id'])
    edge_all = zip(sub_src, sub_dst)
    deta = 0
    if (v,u) in edge_all:
        deta = 1
    else:
        deta = 0
    return deta    
            
####################################################################

sf_links['common_friends'] = sf_links[['src','dst', 'src_all_friends', 'dst_all_friends']].apply(lambda r: common_friends(r['src'], r['dst'],r['src_all_friends'], r['dst_all_friends']))
sf_links['common_bi_friends'] = sf_links[['src','dst', 'src_bi_friends', 'dst_bi_friends']].apply(lambda r: common_friends(r['src'], r['dst'],r['src_bi_friends'], r['dst_bi_friends']))
sf_links['common_in_friends'] = sf_links[['src','dst', 'src_in_friends', 'dst_in_friends']].apply(lambda r: common_friends(r['src'], r['dst'],r['src_in_friends'], r['dst_in_friends']))
sf_links['common_out_friends'] = sf_links[['src','dst', 'src_out_friends', 'dst_out_friends']].apply(lambda r: common_friends(r['src'], r['dst'],r['src_out_friends'], r['dst_out_friends']))

sf_links['total_friends'] = sf_links[['src','dst', 'src_all_friends', 'dst_all_friends']].apply(lambda r: total_friends(r['src'], r['dst'],r['src_all_friends'], r['dst_all_friends']))
sf_links['total_bi_friends'] = sf_links[['src','dst', 'src_bi_friends', 'dst_bi_friends']].apply(lambda r: total_friends(r['src'], r['dst'],r['src_bi_friends'], r['dst_bi_friends']))
sf_links['total_in_friends'] = sf_links[['src','dst', 'src_in_friends', 'dst_in_friends']].apply(lambda r: total_friends(r['src'], r['dst'],r['src_in_friends'], r['dst_in_friends']))
sf_links['total_out_friends'] = sf_links[['src','dst', 'src_out_friends', 'dst_out_friends']].apply(lambda r: total_friends(r['src'], r['dst'],r['src_out_friends'], r['dst_out_friends']))

sf_links['jacc_coef'] = sf_links[['src','dst', 'src_all_friends', 'dst_all_friends']].apply(lambda r: jacc_coef(r['src'], r['dst'],r['src_all_friends'], r['dst_all_friends']))
sf_links['bi_jacc_coef'] = sf_links[['src','dst', 'src_bi_friends', 'dst_bi_friends']].apply(lambda r: jacc_coef(r['src'], r['dst'],r['src_bi_friends'], r['dst_bi_friends']))
sf_links['in_jacc_coef'] = sf_links[['src','dst', 'src_in_friends', 'dst_in_friends']].apply(lambda r: jacc_coef(r['src'], r['dst'],r['src_in_friends'], r['dst_in_friends']))
sf_links['out_jacc_coef'] = sf_links[['src','dst', 'src_out_friends', 'dst_out_friends']].apply(lambda r: jacc_coef(r['src'], r['dst'],r['src_out_friends'], r['dst_out_friends']))

sf_links['transitive_friends'] = sf_links[['src','dst', 'src_all_friends', 'dst_all_friends']].apply(lambda r: transitive_friends(r['src'], r['dst'],r['src_all_friends'], r['dst_all_friends']))
sf_links['bi_transitive_friends'] = sf_links[['src','dst', 'src_bi_friends', 'dst_bi_friends']].apply(lambda r: transitive_friends(r['src'], r['dst'],r['src_bi_friends'], r['dst_bi_friends']))
sf_links['in_transitive_friends'] = sf_links[['src','dst', 'src_in_friends', 'dst_in_friends']].apply(lambda r: transitive_friends(r['src'], r['dst'],r['src_in_friends'], r['dst_in_friends']))
sf_links['out_transitive_friends'] = sf_links[['src','dst', 'src_out_friends', 'dst_out_friends']].apply(lambda r: transitive_friends(r['src'], r['dst'],r['src_out_friends'], r['dst_out_friends']))

sf_links['attachment_score'] = sf_links[['src','dst', 'src_all_friends', 'dst_all_friends']].apply(lambda r: attachment_score(r['src'], r['dst'],r['src_all_friends'], r['dst_all_friends']))
sf_links['bi_attachment_score'] = sf_links[['src','dst', 'src_bi_friends', 'dst_bi_friends']].apply(lambda r: attachment_score(r['src'], r['dst'],r['src_bi_friends'], r['dst_bi_friends']))
sf_links['in_attachment_score'] = sf_links[['src','dst', 'src_in_friends', 'dst_in_friends']].apply(lambda r: attachment_score(r['src'], r['dst'],r['src_in_friends'], r['dst_in_friends']))
sf_links['out_attachment_score'] = sf_links[['src','dst', 'src_out_friends', 'dst_out_friends']].apply(lambda r: attachment_score(r['src'], r['dst'],r['src_out_friends'], r['dst_out_friends']))

sf_links['friends_measure'] = sf_links[['src','dst', 'src_all_friends', 'dst_all_friends']].apply(lambda r: friends_measure(r['src'], r['dst'],r['src_all_friends'], r['dst_all_friends'],g))
sf_links['bi_friends_measure'] = sf_links[['src','dst', 'src_bi_friends', 'dst_bi_friends']].apply(lambda r: friends_measure(r['src'], r['dst'],r['src_bi_friends'], r['dst_bi_friends'],g))
sf_links['in_friends_measure'] = sf_links[['src','dst', 'src_in_friends', 'dst_in_friends']].apply(lambda r: friends_measure(r['src'], r['dst'],r['src_in_friends'], r['dst_in_friends'],g))
sf_links['out_friends_measure'] = sf_links[['src','dst', 'src_out_friends', 'dst_out_friends']].apply(lambda r: friends_measure(r['src'], r['dst'],r['src_out_friends'], r['dst_out_friends'],g))

sf_links['opp_dire_friends'] = sf_links[['src','dst']].apply(lambda r: opp_dire_friends(r['src'], r['dst'], g))

#################################################################### 
'''Edge subgraph features'''
'''edge_nh-subgraph'''
def edge_nh_subgraph(u, v, G):
    u_nh_edge = set(nh_subgraph(u, G))
    v_nh_edge = set(nh_subgraph(v, G))
    nh_edge_all = u_nh_edge | v_nh_edge
    return list(nh_edge_all)          

'''edge_nh-subgraph'''
def edge_nh_subgraph_plus(u, v, G):
    u_nh_edge = set(nh_subgraph_plus(u, G))
    v_nh_edge = set(nh_subgraph_plus(v, G))
    nh_edge_all = u_nh_edge | v_nh_edge
    return list(nh_edge_all)              
 
'''inner-subgraph'''
def inner_subgraph(u,v,u_friends,v_friends,G):
    u_friends = set(u_friends)
    if v in u_friends:
            u_friends.remove(v)
    v_friends = set(v_friends)
    if u in v_friends:
        v_friends.remove(u)
    edge = G.get_edges()
    sub_src = list(edge['__src_id'])
    sub_dst = list(edge['__dst_id'])
    edge_all = zip(sub_src, sub_dst)
    inner_subgraph = []
    for x in u_friends:
        for y in v_friends:
            if (x,y) in edge_all:
                inner_subgraph.append((x,y))
    for x in v_friends:
        for y in u_friends:
            if (x,y) in edge_all:
                inner_subgraph.append((x,y))
    return inner_subgraph      
    
'''scc-edge-subgraph-num'''
def scc_edge_num(u,v,G):
    nh_subg = edge_nh_subgraph(u,v,G)
    scc_num = []
    for x,y in nh_subg:
        if (x,y) and (y,x) in nh_subg:
            scc_num.append((x,y))
    scc_edge_num = len(scc_num)/2
    return scc_edge_num
    
'''wcc-edge-subgraph-num'''
def wcc_edge_num(u,v,G):
    nh_subg = edge_nh_subgraph(u,v,G)
    scc_num = []
    for x,y in nh_subg:
        if (x,y) and (y,x) not in nh_subg:
            scc_num.append((x,y))
    wcc_edge_num = len(scc_num)
    return wcc_edge_num
    
'''scc-edge-subgraph-plus-num'''
def scc_edge_plus_num(u,v,G):
    nh_subg = edge_nh_subgraph_plus(u,v,G)
    scc_num = []
    for x,y in nh_subg:
        if (x,y) and (y,x) in nh_subg:
            scc_num.append((x,y))
    scc_edge_num = len(scc_num)/2
    return scc_edge_num
    
'''scc-inner-subgraph-num'''
def scc_inner_num(u,v,u_friends,v_friends,G):
    nh_subg = inner_subgraph(u,v,u_friends,v_friends,G)
    scc_num = []
    for x,y in nh_subg:
        if (x,y) and (y,x) in nh_subg:
            scc_num.append((x,y))
    scc_edge_num = len(scc_num)/2
    return scc_edge_num
    
'''wcc-inner-subgraph-num'''
def wcc_inner_num(u,v,u_friends,v_friends,G):
    nh_subg = inner_subgraph(u,v,u_friends,v_friends,G)
    scc_num = []
    for x,y in nh_subg:
        if (x,y) and (y,x) not in nh_subg:
            scc_num.append((x,y))
    wcc_edge_num = len(scc_num)
    return wcc_edge_num
    
#################################################################### 
sf_links['edge_nh_subgraph'] = sf_links[['src','dst']].apply(lambda r: edge_nh_subgraph(r['src'], r['dst'], g))
sf_links['edge_nh_subgraph_plus'] = sf_links[['src','dst']].apply(lambda r: edge_nh_subgraph_plus(r['src'], r['dst'], g))

sf_links['inner_subgraph'] = sf_links[['src','dst', 'src_all_friends', 'dst_all_friends']].apply(lambda r: inner_subgraph(r['src'], r['dst'],r['src_all_friends'], r['dst_all_friends'],g))
sf_links['bi_inner_subgraph'] = sf_links[['src','dst', 'src_bi_friends', 'dst_bi_friends']].apply(lambda r: inner_subgraph(r['src'], r['dst'],r['src_bi_friends'], r['dst_bi_friends'],g))
sf_links['in_inner_subgraph'] = sf_links[['src','dst', 'src_in_friends', 'dst_in_friends']].apply(lambda r: inner_subgraph(r['src'], r['dst'],r['src_in_friends'], r['dst_in_friends'],g))
sf_links['out_inner_subgraph'] = sf_links[['src','dst', 'src_out_friends', 'dst_out_friends']].apply(lambda r: inner_subgraph(r['src'], r['dst'],r['src_out_friends'], r['dst_out_friends'],g))

sf_links['scc_edge_num'] = sf_links[['src','dst']].apply(lambda r: scc_edge_num(r['src'], r['dst'], g))
sf_links['wcc_edge_num'] = sf_links[['src','dst']].apply(lambda r: wcc_edge_num(r['src'], r['dst'], g))
sf_links['scc_edge_plus_num'] = sf_links[['src','dst']].apply(lambda r: scc_edge_plus_num(r['src'], r['dst'], g))

sf_links['scc_inner_num'] = sf_links[['src','dst', 'src_out_friends', 'dst_out_friends']].apply(lambda r: scc_inner_num(r['src'], r['dst'],r['src_out_friends'], r['dst_out_friends'],g))
sf_links['bi_scc_inner_num'] = sf_links[['src','dst', 'src_bi_friends', 'dst_bi_friends']].apply(lambda r: scc_inner_num(r['src'], r['dst'],r['src_bi_friends'], r['dst_bi_friends'],g))
sf_links['in_scc_inner_num'] = sf_links[['src','dst', 'src_in_friends', 'dst_in_friends']].apply(lambda r: scc_inner_num(r['src'], r['dst'],r['src_in_friends'], r['dst_in_friends'],g))
sf_links['out_scc_inner_num'] = sf_links[['src','dst', 'src_out_friends', 'dst_out_friends']].apply(lambda r: scc_inner_num(r['src'], r['dst'],r['src_out_friends'], r['dst_out_friends'],g))

sf_links['wcc_inner_num'] = sf_links[['src','dst', 'src_out_friends', 'dst_out_friends']].apply(lambda r: wcc_inner_num(r['src'], r['dst'],r['src_out_friends'], r['dst_out_friends'],g))
sf_links['bi_wcc_inner_num'] = sf_links[['src','dst', 'src_bi_friends', 'dst_bi_friends']].apply(lambda r: wcc_inner_num(r['src'], r['dst'],r['src_bi_friends'], r['dst_bi_friends'],g))
sf_links['in_wcc_inner_num'] = sf_links[['src','dst', 'src_in_friends', 'dst_in_friends']].apply(lambda r: wcc_inner_num(r['src'], r['dst'],r['src_in_friends'], r['dst_in_friends'],g))
sf_links['out_wcc_inner_num'] = sf_links[['src','dst', 'src_out_friends', 'dst_out_friends']].apply(lambda r: wcc_inner_num(r['src'], r['dst'],r['src_out_friends'], r['dst_out_friends'],g))
sf_links.save('sf_link_data')




