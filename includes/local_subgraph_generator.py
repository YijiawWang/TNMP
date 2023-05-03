from copy import deepcopy
import networkx as nx
import matplotlib.pyplot as plt

def Ni_generator(G,center_node,R):
    """Generate the subgraph G_neighborhood(R) of the center node on the corresponding graph G, 
    which satisfies that the shortest path between any two nodes on the boundary of G_neighborhood in G\G_neighborhood is longer than R.
    
    Parameters
    ----------
    G : nx.Graph
        The complete graph.
    center_node : int
        The node id of the center node, it should be an element of list(G.nodes()).
    R : int

    Returns
    -------
    Ni_v : list of int
        The vertex list of G_N.
    Ni_e : list of tuple of int
        The edge list of G_N.
    """
    G_neighborhood = nx.Graph()
    G_environment = deepcopy(G)
    Ni_v = []
    Ni_e = []
    for direct_neighbor_of_i in G.neighbors(center_node):
        Ni_v.append(direct_neighbor_of_i)
        Ni_e.append((center_node,direct_neighbor_of_i))
    G_neighborhood.add_nodes_from(Ni_v)
    G_neighborhood.add_edges_from(Ni_e)
    G_environment.remove_edges_from(Ni_e)
    for r in range(1,R+1):
        focus_layer = 0
        G_neighborhoods,_,_,_,G_environment,_ = neighborhood_grow(G,G_neighborhood,G_environment,r)
        G_neighborhood = G_neighborhoods[-1]
    for node in list(G_neighborhood.nodes()):
        if node not in Ni_v:
            Ni_v.append(node)
    for edge in list(G_neighborhood.edges()):
        if edge not in Ni_e and (edge[1],edge[0]) not in Ni_e:
            Ni_e.append(edge)
    return Ni_v,Ni_e



def neighborhood_grow(G,G_neighborhood0,G_environment,R):
    """Generate the subgraph G_neighborhood(R) by expanding G_neighborhood0 until its boundary satisfies that
    the shortest path between any two nodes on the boundary of G_neighborhood in G\G_neighborhood is longer than R.

    Parameters
    ----------
    G : nx.Graph
        The complete graph.
    G_neighborhood0 : nx.Graph
        The original graph, upon which we obtain the satisfactory G_neighborhood by adding edges and vertices.
    G_environment : nx.Graph
        The complement of G_neighborhood0
    R : int

    Returns
    -------
    G_neighborhoods : list of nx.Graph
        The intermediate G_neighborhoods in the expanding process.
        G_neighborhoods[turn_id] = G_neighborhood at the end of the turn_id-th turn
        A turn refers to a cycle of the two steps of ① determining the boundary nodes ② adding all the short paths between boundary pairs.
    new_nodes : list of list of int 
        The newly added nodes in each turn of the expanding process.
        new_nodes[turn_id] = list(V(G_neighborhood(turn_id))\V(G_neighborhood(turn_id-1)))
    new_edges : list of list of tuple of int
        The newly added edges in each turn of the expanding process.
        new_edges[turn_id] = list(E(G_neighborhood(turn_id))\E(G_neighborhood(turn_id-1)))
    boundaries : list of list of int
        The boundaries of the intermediate G_neighborhoods.
        boundaries[turn_id] = boundary nodes list of G_neighborhoods[turn_id]
    G_environment : nx.Graph
        The complement of the final G_neighborhood
    turn : int
        The number of turns required to obtain the final G_neighborhood.
    """
    stop = 0
    turn = 0
    G_neighborhood = deepcopy(G_neighborhood0)
    G_neighborhoods = []
    new_nodes = []
    new_edges = []
    boundaries = []
    while stop == 0:
        stop = 1
        turn += 1
        new_nodes_per_turn = []
        new_edges_per_turn = []
        boundary = []
        for node in list(G_neighborhood.nodes()):
            for noden in G.neighbors(node):
                if not G_neighborhood.has_edge(node,noden):
                    boundary.append(node)
                    break
        if turn != 1:
            boundaries.append(boundary)
        l_b = len(boundary)
        for index1 in range(l_b):
            for index2 in range(index1+1,l_b):
                end1 = boundary[index1]
                end2 = boundary[index2]
                if nx.has_path(G_environment, end1, end2) == True:
                    shortest_paths = list(nx.all_shortest_paths(G_environment, end1, end2))
                else:
                    shortest_paths = []
                while len(shortest_paths) != 0 and len(shortest_paths[0]) <= R+1 :
                    stop = 0
                    for shortest_path in shortest_paths:
                        for node_id in range(len(shortest_path)-1):
                            if not G_neighborhood.has_edge(shortest_path[node_id],shortest_path[node_id+1]):
                                if not G_neighborhood.has_node(shortest_path[node_id]):
                                    new_nodes_per_turn.append(shortest_path[node_id])
                                if not G_neighborhood.has_node(shortest_path[node_id + 1]):
                                    new_nodes_per_turn.append(shortest_path[node_id + 1])
                                new_edges_per_turn.append((shortest_path[node_id],shortest_path[node_id+1]))
                                G_neighborhood.add_edge(shortest_path[node_id],shortest_path[node_id+1])
                                G_environment.remove_edge(shortest_path[node_id],shortest_path[node_id+1])
                    if nx.has_path(G_environment, end1, end2) == True:
                        shortest_paths = list(nx.all_shortest_paths(G_environment, end1, end2))
                    else:
                        shortest_paths = []
        new_nodes.append(new_nodes_per_turn)
        new_edges.append(new_edges_per_turn)
        G_neighborhoods.append(deepcopy(G_neighborhood))    
    return G_neighborhoods,new_nodes,new_edges,boundaries,G_environment,turn



def cavity_subgraph_generator(Ne,a,i):
    """Generate the subgraph G_C_{a → i} of the cavity sub-network, which is the edge induced subgraph of E(G_N_a)\E(G_N_i)
    
    Parameters
    ----------
    Ne : list of list of tuple of int
        The list of the edge lists of all the G_N, Ne[i] = list(E(G_N_i)).
    a : int
        The node id of the boundary node, it should be an element of list(G.nodes()).
    i : int
        The node id of the center node, it should be an element of list(G.nodes()).

    Returns
    -------
    G_cavity :  nx.Graph
        The subgraph G_C_{a → i}.
    """
    G_neighborhood_a = nx.Graph()
    G_neighborhood_i = nx.Graph()
    G_cavity = nx.Graph()
    G_neighborhood_a.add_edges_from(Ne[a])
    G_neighborhood_i.add_edges_from(Ne[i])
    Nc_e = []
    for node1,node2 in Ne[a]:
        if not G_neighborhood_i.has_edge(node1,node2):
            Nc_e.append((node1,node2))
    G_cavity.add_edges_from(Nc_e)
    
    return G_cavity