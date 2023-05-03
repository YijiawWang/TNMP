import numpy as np
import networkx as nx


def read_model(G_file, J_file, h_file):
    """Read the model parameters from the given files stored in "/constants" and convert them to the data type required for subsequent calculations.

    Parameters
    ----------
    G_file : str
        The name of the file storing the graph structure upon which the model is defined, and the corresponding file should be in the gexf format.
    J_file : str
        The name of the file storing the coupling constants and the corresponding file should be in the csv format. 
        There should be |E(G)| lines of the J_file, each of which is in the format of (i, j, J_ij) for J_ij acting on edge (i, j) of G.
    h_file : str
        The name of the file storing the external fields and the corresponding file should be in the csv format.
        There should be |V(G)| lines of the h_file, each of which should be a float representing the field h_i on node i.

    Returns
    -------
    G : nx.Graph
        Contains the information of the graph structure.
    J : array
        The coupling constants array with a shape of [n, n], J[i][j] = J_ij for every edge (i, j) of G and in all other positions of J are filled with 0.
    h : array
        The field array with a shape of [n] and h[i] = h_i.
    """
    G0 = nx.read_gexf('constants/{}.gexf'.format(G_file))
    n = G0.number_of_nodes()
    G = nx.Graph()
    for edge in list(G0.edges()):
        G.add_edge(int(edge[0]), int(edge[1]))

    data_edge = np.loadtxt(
        open("constants/{}.csv".format(J_file), "rb"), delimiter=",")
    J = np.zeros([n, n])
    for row in data_edge:
        J[int(row[0])][int(row[1])] = row[2]
        J[int(row[1])][int(row[0])] = row[2]

    data_field = np.loadtxt(
        open("constants/{}.csv".format(h_file), "rb"), delimiter=",")
    h = np.zeros([n, ])
    for i in range(len(data_field)):
        h[i] = data_field[i]

    return G, J, h

