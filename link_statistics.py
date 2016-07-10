# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 13:12:03 2016

@author: hh
"""

######################################################################
''' vertex features  '''

''' all_friends '''
def all_friends(in_friends, out_friends):
    in_f = set(in_friends)
    out_f = set(out_friends)
    all_friend = in_f | out_f
    return list(all_friend)
    
'''all_friends+'''
def all_friends_plus(ids, in_friends, out_friends):
    allf = all_friends(in_friends, out_friends)
    all_friend_plus = allf.append(ids)
    return all_friend_plus

'''nh-subgraph'''
def nh_subgraph(idl, G):
    subg = G.get_neighborhood(ids = list(idl), radius = 2, full_subgraph = True)
    sub_edge = subg.get_edges()
    sub_src = list(sub_edge['__src_id'])
    sub_dst = list(sub_edge['__dst_id'])
    nh_sub = zip(sub_src, sub_dst)
    for i in nh_sub:
        if idl in i:
            nh_sub.remove(i)
    return nh_sub
    
'''nh-subgraph+'''    
def nh_subgraph_plus(idl, G):
    subg = G.get_neighborhood(ids = list(idl), radius = 2, full_subgraph = True)
    sub_edge = subg.get_edges()
    sub_src = list(sub_edge['__src_id'])
    sub_dst = list(sub_edge['__dst_id'])
    nh_sub = zip(sub_src, sub_dst)
    return nh_sub

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
    num_bi_friends = len(in_friends & out_friends)
    if  num_all_friends == 0:
        return 0
    return num_bi_friends/float(num_all_friends)
    
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
''' edge features  '''    
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