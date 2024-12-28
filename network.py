import pickle as pk
import networkx as nx
import numpy as np
from random import sample
import matplotlib.pyplot as plt
import sys
from pycirclize import Circos
import pandas as pd

iters = 0
otus = []
with open("data/relative_abun_trans.csv", "r") as handle:
    for i in handle.readlines():
        otus.append(i.split(",")[0])
otus = otus[1:]

regulate_matrix = np.load("data/adj_mat.npy")

G = nx.Graph()

transverse = {
    "k__":"kingdom",
    "p__":"phylum",
    "c__":"class",
    "o__":"order",
    "f__":"family",
    "g__":"genus"
    #"s__":"species"
}

nodes = []
phylums = []

for i in range(0, len(otus)):
    otu = otus[i]
    node_name = ""
    for key in transverse:
        if (key in otu) and (otu.split(key)[1].split(";")[0] != ""):
            node_name = otu.split(key)[1].split(";")[0]
    if ("s__" in otu) and (otu.split("s__")[1] != ""):
        node_name = otu.split("s__")[1]
    try:
        if otu.split("p__")[1].split(";")[0] == "":
            phylums.append(otu.split("k__")[1].split(";")[0])
        else:
            phylums.append(otu.split("p__")[1].split(";")[0])
    except:
        phylums.append(otu.split("k__")[1].split(";")[0])
    nodes.append(node_name)
print(nodes)
G.add_nodes_from(nodes)
weights = []
for i in range(0, len(regulate_matrix)):
    for j in range(0, len(regulate_matrix)):
        if regulate_matrix[i][j] != 0:
            weights.append((nodes[i], nodes[j], abs(regulate_matrix[i][j])*10))
G.add_weighted_edges_from(weights)
print(weights)
with open("checkpoints/network_edge.csv","w")as f:
    f.write("from,to,value\n")
    for i in weights:
        f.write("{from_},{to},{value}\n".format(from_=i[0], to=i[1], value=i[2]))
with open("checkpoints/network_node.csv","w")as f:
    f.write("from,to\n")
    for i in range(0, len(nodes)):
        f.write("{from_},{to}\n".format(from_=phylums[i], to=nodes[i]))
nx.write_gexf(G,'checkpoints/network.gexf')

matrix_df = pd.DataFrame(regulate_matrix, index=phylums, columns=phylums)
circos = Circos.initialize_from_matrix(
    matrix_df,
    space=3,
    r_lim=(90, 100),
    cmap="tab10",
    label_kws=dict(r=100, size=12, color="black"),
)

print(matrix_df)
fig = circos.plotfig()
fig.savefig("circos.png")