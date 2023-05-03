import networkx as nx


def tensor_network_map(G_sub, G_focus, G, center_node, open):
    """Map G_sub and part of the environment graph which is adjacent to it to the corresponding tensor network in graph format.

    Parameters
    ----------
    G_sub : nx.Graph 
    G_focus : nx.Graph 
        The complete local graph whose corresponding tensor network is showed in the figure.
    G : nx.Graph
        The complete graph.
    center_node : int
        The node id of the center node. 
        The center node is always an internal bond regardless of whether it has edges connected to vertices outside of G_sub.
    open : Bool
        True if there is an open leg on the center node and otherwise False.

    Returns
    -------
    G_tn : nx.Graph
        The tensor network in graph format mapped from G_focus.
    position_T : dict[int,array]
        The positions of the nodes in the figure. 
        The key is the node id and the value is its position, which is an array with a shape of [2] corresponding to x and y coordinates.
    square_nodes_focus : list of int
        The node ids of the Boltzmann matrices in the corresponding tensor network of G_focus.
    square_nodes_neighborhood : list of int
        The node ids of the Boltzmann matrices in the corresponding tensor network of G_sub.
    boundary_bonds : list of int
        The node ids of the boundary copy tensors in the corresponding tensor network of G_sub.
    boundary_nodes : list of int
        The node ids of the message vectors in the corresponding tensor network of G_sub.
    internal_bonds : list of int
        The node ids of the internal copy tensors in the corresponding tensor network of G_sub.
    internal_nodes : list of int
        The node ids of the field vectors in the corresponding tensor network of G_sub.
    environment_bonds : list of int
        The node ids of the copy tensors outside the corresponding tensor network of G_sub.
    environment_nodes : list of int
        The node ids of the field vectors outside the corresponding tensor network of G_sub.
    open_id : int
        The node id of the other end of the open leg on the center node.
    """
    G_tn = nx.Graph()
    n = G.number_of_nodes()
    square_nodes_focus = []
    square_nodes_neighborhood = []
    for edge in list(G_focus.edges()):
        if edge in list(G.edges()):
            J_id = 10*n + list(G.edges()).index(edge)
        else:
            J_id = 10*n + list(G.edges()).index((edge[1], edge[0]))
        G_tn.add_edge(J_id, edge[0])
        G_tn.add_edge(J_id, edge[1])
        square_nodes_focus.append(J_id)
        if G_sub.has_edge(edge[0],edge[1]):
            square_nodes_neighborhood.append(J_id)
    boundary_bonds = []
    for node in list(G_sub.nodes()):
        for noden in G.neighbors(node):
            if not G_sub.has_edge(node, noden):
                boundary_bonds.append(node)
                break
    internal_bonds = list(set(list(G_sub.nodes()))-set(boundary_bonds))
    environment_bonds = list(set(list(G_focus.nodes())) - set(list(G_sub.nodes())))
    boundary_nodes = []
    for boundary_bond in boundary_bonds:
        if boundary_bond != center_node:
            cid = 100*n+G.number_of_edges()+boundary_bond
            G_tn.add_edge(cid, boundary_bond)
            boundary_nodes.append(cid)
    internal_nodes = []
    for internal_bond in internal_bonds:
        fid = 100*n+G.number_of_edges()+internal_bond
        G_tn.add_edge(fid, internal_bond)
        internal_nodes.append(fid)
    if center_node not in internal_bonds and center_node != False:
        fid = 100*n+G.number_of_edges()+center_node
        G_tn.add_edge(fid, center_node)
        internal_nodes.append(fid)
    environment_nodes = []
    for environment_bond in environment_bonds:
        fid = 100*n+G.number_of_edges()+environment_bond
        G_tn.add_edge(fid, environment_bond)
        environment_nodes.append(fid)
    if open != False:
        open_id = 200*n+G.number_of_edges()
        G_tn.add_edge(open_id, center_node)
    else:
        open_id = False
    position_T = nx.kamada_kawai_layout(G_tn)

    return G_tn,position_T,square_nodes_focus,square_nodes_neighborhood,boundary_bonds,boundary_nodes,internal_bonds,internal_nodes,environment_bonds,environment_nodes,open_id


