# -*- coding: utf-8 -*-
"""
calculate the mic of statistics

"""


import graphlab as gl
import numpy as np
from minepy import MINE
import matplotlib.pyplot as plt

sf_links = gl.load_sframe('sf_link_data')
link_features_list = ['common_friends', 'common_in_friends', 'common_out_friends', 'common_bi_friends', 'total_friends', 'total_in_friends', 'total_out_friends', 'total_bi_friends',
'jacc_coef', 'bi_jacc_coef', 'in_jacc_coef', 'out_jacc_coef']
fs = [c for c in sf_links.column_names() if "coedge" in c] #+ [c for c in sf_links.column_names() if "measure" in c] + link_features_list

label = []
la = ['label']
#la = []
res = []
for i in range(len(fs)):
    label.append(list(sf_links['class']))
    #la.append('class')
    res.append(list(sf_links[fs[i]]))

cm = []
for i in range(len(fs)):
    tmp = []
    for j in range(len(fs)):
        m = MINE()
        m.compute_score(res[i], label[0])
        tmp.append(m.mic())
        cm.append(tmp)

def plot_confusion_matrix(cm, title, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(fs))
    tt = np.arange(1)
#    plt.xticks(tick_marks, la, rotation=45)
    plt.xticks(tt, la, rotation=45)
    plt.yticks(tick_marks, fs)
    plt.tight_layout()

plot_confusion_matrix(cm, title='mic')
plt.show()

