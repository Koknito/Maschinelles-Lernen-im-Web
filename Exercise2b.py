import math
from random import random
import numpy
import numpy as np
import scipy
import networkx as nx
G = nx.Graph().edges

node_count = 100
probability = 0.1

erdos = nx.erdos_renyi_graph(node_count, probability)
barabasi = nx.barabasi_albert_graph(node_count, round((node_count-1)*probability/2))

def output(res, erdos, barabsi):
    print()
    print("Erdös and Renyi, " , res, ":", erdos)
    print("Barabasi and Albert, ", res, ":", barabsi)
output("Edgecount", len(erdos.edges), len(barabasi.edges))
output("Pagerank", nx.pagerank(erdos), nx.pagerank(barabasi))
erdos_pageranklist = sorted(list(nx.pagerank(erdos).values()))
barabasi_pageranklist = sorted(list(nx.pagerank(barabasi).values()))
output("(Descending) Sorted Pagerank", sorted(list(nx.pagerank(erdos).values()), reverse=True), sorted(list(nx.pagerank(barabasi).values()), reverse=True))
output("(Ascending) Sorted Pagerank", sorted(list(nx.pagerank(erdos).values())), sorted(list(nx.pagerank(barabasi).values())))
output("First order moment", scipy.stats.moment(erdos_pageranklist, 1), scipy.stats.moment(barabasi_pageranklist, 1))
output("Second order pagerank", scipy.stats.moment(erdos_pageranklist, order=2), scipy.stats.moment(barabasi_pageranklist, order=2))
output("Third order pagerank", scipy.stats.moment(erdos_pageranklist, order=3), scipy.stats.moment(barabasi_pageranklist, order=3))


