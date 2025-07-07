from turtle import position
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from local_subgraph_generator import cavity_subgraph_generator
from graph_tensor_network_map import tensor_network_map
from copy import deepcopy
import seaborn as sns
from IPython import display


def get_layer_environment_node(G, layer, center_node):
    """Generate the subgraph containing all vertices with a distance less than layer from the center_node and the edges between them.

    Parameters
    ----------
    G : nx.Graph 
        The complete graph.
    layer: int
        The number of layers of neighbors included in the subgraph.
    center_node : int
        The node_id of the center node.
        
    Returns
    -------
    G_focus :  nx.Graph
        The result subgraph.
    """
    G_focus = nx.Graph()
    layers = [[center_node]]
    history_nodes = [center_node]
    Ni_v = [center_node]
    Ni_e = []
    for layer_id in range(layer):
        neighbor_list = []
        history_nodes0 = deepcopy(history_nodes)
        for node1 in layers[layer_id]:
            for node2 in list(G.neighbors(node1)):
                if node2 not in history_nodes:
                    neighbor_list.append(node2)
                    history_nodes.append(node2)
        for node_n in neighbor_list:
            Ni_e_new = []
            for node_nn in list(G.neighbors(node_n)):
                if node_nn in history_nodes0:
                    Ni_e_new.append((node_n, node_nn))
            Ni_v = Ni_v + [node_n]
            Ni_e = Ni_e + Ni_e_new
            history_nodes0.append(node_n)
        layers.append(neighbor_list)
    G_focus.add_nodes_from(Ni_v)
    G_focus.add_edges_from(Ni_e)
    return G_focus



def get_layer_environment_gsub(G, layer, G_sub):
    """Generate the subgraph containing all vertices with a distance less than layer from the vertices in G_sub and the edges between them.

    Parameters
    ----------
    G : nx.Graph 
        The complete graph.
    layer: int
        The number of layers of neighbors included in the subgraph.
    G_sub : nx.Graph
     
    Returns
    -------
    G_focus :  nx.Graph
        The result subgraph.
    """
    G_focus = nx.Graph()
    layers = [deepcopy(list(G_sub.nodes()))]
    history_nodes = deepcopy(list(G_sub.nodes()))
    Ni_v = deepcopy(list(G_sub.nodes()))
    Ni_e = deepcopy(list(G_sub.edges()))
    for layer_id in range(layer):
        neighbor_list = []
        history_nodes0 = deepcopy(history_nodes)
        for node1 in layers[layer_id]:
            for node2 in list(G.neighbors(node1)):
                if node2 not in history_nodes:
                    neighbor_list.append(node2)
                    history_nodes.append(node2)
        for node_n in neighbor_list:
            Ni_e_new = []
            for node_nn in list(G.neighbors(node_n)):
                if node_nn in history_nodes0:
                    Ni_e_new.append((node_n, node_nn))
            Ni_v = Ni_v + [node_n]
            Ni_e = Ni_e + Ni_e_new
            history_nodes0.append(node_n)
        layers.append(neighbor_list)
    G_focus.add_nodes_from(Ni_v)
    G_focus.add_edges_from(Ni_e)
    return G_focus



def neighborhood_show(G_sub, G_focus, center_node, new_nodes, new_edges, boundary, title, position, ax):
    """Plot G_N_i and part of the environment graph which is adjacent to it.

    Parameters
    ----------
    G_sub : nx.Graph
        The corresponding graph of the neighborhood of the center node.
    G_focus : nx.Graph
        The complete local graph showed in the figure.
    center_node : int
        The node id of the center node i.
    new_nodes : list of int 
        The newly added nodes to G_N_i(R-1) or to G_N_i(R) of the last turn.
    new_edges : list of tuple of int
        The newly added nodes to G_N_i(R-1) or to G_N_i(R) of the last turn.
    boundary : list of int
        The boundary nodes of G_N_i.
    title : str
        The title of the figure.
    position : dict[int,array]
        The positions of the nodes in the figure. 
        The key is the node id and the value is its position, which is an array with a shape of [2] corresponding to x and y coordinates.
    ax : AxesSubplot
        The coordinates in the entire figure when the drawn figure appears as a subfigure.
    """
    
    node_color = ['k'] * len(G_focus.nodes())
    edge_color = ['k'] * len(G_focus.edges())
    node_sizes = [400] * len(G_focus.nodes())
    node_sizes[list(G_focus.nodes()).index(center_node)] = 600
    node_shape = ['o'] * len(G_focus.nodes())
    for node in boundary:
        node_shape[list(G_focus.nodes()).index(node)] = 'h'
    for node in list(G_sub.nodes()):
        index_node = list(G_focus.nodes()).index(node)
        node_color[index_node] = 'g'
    for edge in list(G_sub.edges()):
        if edge in list(G_focus.edges()):
            index_edge = list(G_focus.edges()).index(edge)
        else:
            index_edge = list(G_focus.edges()).index((edge[1], edge[0]))
        edge_color[index_edge] = 'g'
    for node in new_nodes:
        node_sizes[list(G_focus.nodes()).index(node)] = 800
        node_color[list(G_focus.nodes()).index(node)] = 'orange'
    node_color[list(G_focus.nodes()).index(center_node)] = 'r'
    
    for edge in new_edges:
        if edge in list(G_focus.edges()):
            edge_color[list(G_focus.edges()).index(edge)] = 'orange'
        else:
            edge_color[list(G_focus.edges()).index((edge[1],edge[0]))] = 'orange'
    
    graph_draw(G_focus,position,node_sizes,3,3,'w',node_color,edge_color,node_shape,ax,title)



def cavity_show(G_sub, G_focus,  boundary_node, center_node, title, position, ax):
    """Plot G_C_{a → i} and part of the environment graph which is adjacent to it.
    
    Parameters
    ----------
    G_sub : nx.Graph
        The corresponding graph of the cavity network C_{a → i}.
    G_focus : nx.Graph
        The complete local graph showed in the figure.
    boundary_node : int
        The node id of the boundary node a.
    center_node : int
        The node id of the center node i.
    title : str
        The title of the figure.
    position : dict[int,array]
        The positions of the nodes in the figure. 
        The key is the node id and the value is its position, which is an array with a shape of [2] corresponding to x and y coordinates.
    ax : AxesSubplot
        The coordinates in the entire figure when the drawn figure appears as a subfigure.
    """
    
    node_color = ['k'] * len(G_focus.nodes())
    edge_color = ['k'] * len(G_focus.edges())
    for node in list(G_sub.nodes()):
        index_node = list(G_focus.nodes()).index(node)
        node_color[index_node] = 'g'
    for edge in list(G_sub.edges()):
        if edge in list(G_focus.edges()):
            index_edge = list(G_focus.edges()).index(edge)
        else:
            index_edge = list(G_focus.edges()).index((edge[1], edge[0]))
        edge_color[index_edge] = 'g'
    
    node_color[list(G_focus.nodes()).index(center_node)] = 'r'
    node_color[list(G_focus.nodes()).index(boundary_node)] = 'b'
    
    graph_draw(G_focus,position,[150]*G_focus.number_of_nodes(),3,3,'w',node_color,edge_color,['o']*G_focus.number_of_nodes(),ax,title)
    


def tensor_network_show(G_sub, G, G_focus, center_node, title, ax):
    """Plot the corresponding tensor network of G_sub and part of the environment tensor network which is adjacent to it.
    
    Parameters
    ----------
    G_sub : nx.Graph 
        The corresponding graph of the colored sub network.
    G : nx.Graph
        The complete graph of the model.
    G_focus : nx.Graph
        The corresponding graph of the complete local graph showed in the figure.
    center_node : int
        The node id of the center node i.
    title : str
        The title of the figure.
    ax : AxesSubplot
        The coordinates in the entire figure when the drawn figure appears as a subfigure.

    Returns
    -------
    position : dict[int,array]
        The positions of the nodes in the figure. 
        The key is the node id and the value is its position, which is an array with a shape of [2] corresponding to x and y coordinates.
    """
    G_tn,position_T,square_nodes_focus,square_nodes_neighborhood,boundary_bonds,boundary_nodes,internal_bonds,internal_nodes,environment_bonds,environment_nodes,open_id = tensor_network_map(G_sub, G_focus, G, center_node, open)
    node_size = 150
    node_color = ['gray'] * len(G_tn.nodes())
    edge_color = ['gray'] * len(G_tn.edges())
    node_shapes = ['o'] * len(G_tn.nodes())
    node_size = [node_size] * len(G_tn.nodes())
    styles = ['dashed']* len(G_tn.edges())

    for node in square_nodes_focus:
        node_shapes[list(G_tn.nodes()).index(node)] = 's'
    for node in square_nodes_neighborhood:
        node_color[list(G_tn.nodes()).index(node)] = 'green'
        node_shapes[list(G_tn.nodes()).index(node)] = 's'
    for boundary_bond in boundary_bonds:
        node_color[list(G_tn.nodes()).index(boundary_bond)] = 'pink'
    for internal_bond in internal_bonds:
        node_color[list(G_tn.nodes()).index(internal_bond)] = 'blue'
    node_color[list(G_tn.nodes()).index(center_node)] = 'red'
    for node in boundary_nodes:
        node_color[list(G_tn.nodes()).index(node)] = 'purple'
        node_shapes[list(G_tn.nodes()).index(node)] = 'd'
    for node in internal_nodes:
        node_color[list(G_tn.nodes()).index(node)] = 'orange'
        node_shapes[list(G_tn.nodes()).index(node)] = 'h'
    for node in environment_nodes:
        node_shapes[list(G_tn.nodes()).index(node)] = 'h'
    if (open_id, center_node) in list(G_tn.edges()):
        edge_color[list(G_tn.edges()).index((open_id, center_node))] = 'red'
        styles[list(G_tn.edges()).index((open_id, center_node))] = 'solid'
    else:
        edge_color[list(G_tn.edges()).index((center_node, open_id))] = 'red'
        styles[list(G_tn.edges()).index((center_node, open_id))] = 'solid'
    for edge in list(G_tn.edges()):
        if edge[0] in internal_nodes+internal_bonds+boundary_nodes+boundary_bonds+square_nodes_neighborhood and edge[1] in internal_nodes+internal_bonds+boundary_nodes+boundary_bonds+square_nodes_neighborhood:
            if edge in list(G_tn.edges()):
                styles[list(G_tn.edges()).index(edge)] = 'solid'
                edge_color[list(G_tn.edges()).index(edge)] = 'k'
            else:
                styles[list(G_tn.edges()).index((edge[1],edge[0]))] = 'solid'
                edge_color[list(G_tn.edges()).index(edge)] = 'k'
    node_color[list(G_tn.nodes()).index(open_id)] = 'red'
    node_size[list(G_tn.nodes()).index(open_id)] = 0
    # plt.subplot(axs[1])
    #plt.axis('off')
    position_T = nx.kamada_kawai_layout(G_tn)
    for node in list(G_tn.nodes()):
        pos_node = {}
        pos_node[node] = position_T[node]
        nx.draw_networkx_nodes(
            G_tn, pos=pos_node, nodelist=[node],
            node_size=[node_size[list(G_tn.nodes()).index(node)]],
            node_color='w',
            edgecolors=[node_color[list(G_tn.nodes()).index(node)]],
            node_shape=node_shapes[list(G_tn.nodes()).index(node)],
            linewidths=3, ax=ax)
    nx.draw_networkx_edges(G_tn, pos=position_T, edge_color=edge_color, width=3, style=styles, ax=ax)
    
    ax.set_title(title, fontsize=30,pad=-30)
    
    return position_T



def process_animation_show(G, Ne, boundary_node, center_node, pause_time):
    """Animate the process of the calculation of m_{a → i}.
    
    Parameters
    ----------
    G : nx.Graph
        The complete graph.
    Ne : list of list of tuple of int
        The list of the edge lists of all the G_N, Ne[i] = list(E(G_N_i)).
    boundary_node : int
        The node id of the boundary node a.
    center_node : int
        The node id of the center node i.
    pause_time : float
        The pause time for each frame.
    """
    figsize = [16, 12]
    G0 = get_layer_environment_node(G, 1, center_node)
    G_focus = nx.Graph()
    for edge in list(G0.edges()):
        if not G_focus.has_edge(edge[0], edge[1]):
            G_focus.add_edge(edge[0], edge[1])
    G_cavity = cavity_subgraph_generator(Ne, boundary_node, center_node)
    for edge in list(G_cavity.edges()):
        if not G_focus.has_edge(edge[0], edge[1]):
            G_focus.add_edge(edge[0], edge[1])
    boundary = []
    for node in list(G_cavity.nodes()):
        for noden in G.neighbors(node):
            if not G_cavity.has_edge(node, noden):
                if node != boundary_node:
                    boundary.append(node)
                break
    cavity_Gs = []
    for sub_boundary_node in boundary:
        G_sub_cavity = cavity_subgraph_generator(
            Ne, sub_boundary_node, boundary_node)
        cavity_Gs.append(G_sub_cavity)
        for edge in list(G_sub_cavity.edges()):
            if not G_focus.has_edge(edge[0], edge[1]):
                G_focus.add_edge(edge[0], edge[1])
    
    T_focus,position_T,square_nodes,_,boundary_bonds,boundary_nodes_focus,internal_bonds,internal_nodes,environment_bonds,environment_nodes,open_id = tensor_network_map(G_focus, G_focus, G, center_node,True)
    T_cavity,_,_,_,_,boundary_nodes_cavity,_,_,_,_,_ = tensor_network_map(G_cavity, G_cavity, G, boundary_node ,False)
    
    cavity_Ts = []
    boundaries_sub_cavity = []
    for bid in range(len(cavity_Gs)):
        G_sub_cavity = cavity_Gs[bid]
        T_sub_cavity,_,_,_,_,boundary_sub_cavity,_,_,_,_,_ = tensor_network_map(G_sub_cavity, G_sub_cavity, G, boundary[bid], False)
        cavity_Ts.append(T_sub_cavity)
        boundaries_sub_cavity.append(boundary_sub_cavity)

    node_color = np.zeros(shape=(len(T_focus.nodes()), 3))
    node_shape = ['o'] * len(T_focus.nodes())
    node_size = [300] * len(T_focus.nodes())
    edge_color = ['k'] * len(T_focus.edges())
    colors = sns.color_palette("hls", len(boundary)+1)

    for node in square_nodes:
        node_shape[list(T_focus.nodes()).index(node)] = 's'
    for node in internal_nodes+environment_nodes:
        node_shape[list(T_focus.nodes()).index(node)] = 'h'
    for node in boundary_nodes_focus:
        node_shape[list(T_focus.nodes()).index(node)] = 'd'

    node_size[list(T_focus.nodes()).index(open_id)] = 0
    if (center_node,open_id) in list(T_focus.edges()):
        edge_color[list(T_focus.edges()).index((center_node,open_id))] = np.array([1, 0, 0])
    else:
        edge_color[list(T_focus.edges()).index((open_id,center_node))] = np.array([1, 0, 0])

    for node in list(T_cavity.nodes()):
        node_color[list(T_focus.nodes()).index(node)] = colors[0]
    for edge in list(T_cavity.edges()):
        if edge in list(T_focus.edges()):
            edge_color[list(T_focus.edges()).index(edge)] = colors[0]
        else:
            edge_color[list(T_focus.edges()).index((edge[1], edge[0]))] = colors[0]
    node_color[list(T_focus.nodes()).index(center_node)] = np.array([1, 0, 0])
    node_color[list(T_focus.nodes()).index(boundary_node)] = np.array([0, 0, 1])
    plt.figure(figsize=(figsize[0], figsize[1]))
    plt.clf()
    graph_draw(T_focus, pos=position_T, node_size=node_size, linewidths=3,width=3, node_color='w', edgecolors=node_color, edge_color=edge_color, node_shape=node_shape)
    plt.show()
    display.clear_output(wait=True)
    plt.pause(pause_time)
    for bid in range(len(boundary)):
        for node in list(cavity_Ts[bid].nodes()):
            if node not in boundary:
                node_color[list(T_focus.nodes()).index(node)] = colors[bid+1]
        for edge in list(cavity_Ts[bid].edges()):
            if edge in list(T_focus.edges()):
                edge_color[list(T_focus.edges()).index(edge)] = colors[bid+1]
            else:
                edge_color[list(T_focus.edges()).index((edge[1], edge[0]))] = colors[bid+1]
        for node in boundaries_sub_cavity[bid]:
            node_shape[list(T_focus.nodes()).index(node)] = 'd'
        plt.figure(figsize=(figsize[0], figsize[1]))
        plt.clf()
        graph_draw(T_focus, pos=nx.kamada_kawai_layout(T_focus), node_size=node_size, linewidths=3,width=3, node_color='w', edgecolors=node_color, edge_color=edge_color, node_shape=node_shape)
        plt.show()
        display.clear_output(wait=True)
        plt.pause(pause_time)
        for node in list(cavity_Ts[bid].nodes()):
            if len(list(T_focus.neighbors(node))) == 1 and boundary[bid] in list(T_focus.neighbors(node)):
                node_shape[list(T_focus.nodes()).index(node)] = 'd'
                message_vector_id = node
            elif node not in boundary:
                node_color[list(T_focus.nodes()).index(node)] = np.array([1, 1, 1])
                node_size[list(T_focus.nodes()).index(node)] = 0
        for edge in list(cavity_Ts[bid].edges()):
            if edge in list(T_focus.edges()) and edge[0] != message_vector_id and edge[1] != message_vector_id:
                edge_color[list(T_focus.edges()).index(edge)] = np.array([1, 1, 1])
            elif edge[0] != message_vector_id and edge[1] != message_vector_id:
                edge_color[list(T_focus.edges()).index((edge[1], edge[0]))] = np.array([1, 1, 1])
        plt.figure(figsize=(figsize[0], figsize[1]))
        plt.clf()
        graph_draw(T_focus, pos=position_T, node_size=node_size, linewidths=3,width=3, node_color='w', edgecolors=node_color, edge_color=edge_color, node_shape=node_shape)
        plt.show()
        display.clear_output(wait=True)
        plt.pause(pause_time)

    
    for node in list(T_cavity.nodes()):
        if node != boundary_node:
            node_color[list(T_focus.nodes()).index(node)] = colors[0]
    for edge in list(T_cavity.edges()):
        if edge in list(T_focus.edges()):
            edge_color[list(T_focus.edges()).index(edge)] = colors[0]
        else:
            edge_color[list(T_focus.edges()).index((edge[1], edge[0]))] = colors[0]
    plt.figure(figsize=(figsize[0], figsize[1]))
    plt.clf()
    graph_draw(T_focus, pos=position_T, node_size=node_size, linewidths=3,width=3, node_color='w', edgecolors=node_color, edge_color=edge_color, node_shape=node_shape)
    plt.show()
    display.clear_output(wait=True)
    plt.pause(pause_time)
    
    for node in list(T_cavity.nodes()):
        if len(list(T_focus.neighbors(node))) == 1 and boundary_node in list(T_focus.neighbors(node)):
            node_shape[list(T_focus.nodes()).index(node)] = 'd'
            position_T[node] = deepcopy(position_T[boundary_node])
            position_T[node][0] -= 0.2
            message_vector_id = node
        elif node != boundary_node:
            node_color[list(T_focus.nodes()).index(node)] = np.array([1,1,1])
            node_size[list(T_focus.nodes()).index(node)] = 0
    for edge in list(T_cavity.edges()):
        if edge in list(T_focus.edges()) and edge[0] != message_vector_id and edge[1] != message_vector_id:
            edge_color[list(T_focus.edges()).index(edge)] = np.array([1,1,1])
        elif edge[0] != message_vector_id and edge[1] != message_vector_id:
            edge_color[list(T_focus.edges()).index((edge[1],edge[0]))] = np.array([1,1,1])
    plt.figure(figsize=(figsize[0],figsize[1]))
    plt.clf()
    graph_draw(T_focus, pos=position_T, node_size=node_size, linewidths=3,width=3, node_color='w', edgecolors=node_color, edge_color = edge_color, node_shape=node_shape)
    plt.show()
    display.clear_output(wait=True)
    plt.pause(pause_time)
    


def graph_draw(G, pos, node_size, linewidths, width, node_color, edgecolors, edge_color, node_shape,ax=None,title=None):
    """Plot G in customized shapes, colors, and styles.
    
    Parameters
    ----------
    G : nx.Graph
    pos : dict[int,array]
        The positions of the nodes in the figure. 
        The key is the node id and the value is its position, which is an array with a shape of [2] corresponding to x and y coordinates.
    node_size : list of int
        node_size[i] = size of the i-th node in list(G.nodes()).
    linewidths : list of int
        linewidths[i] = border width of the i-th node in list(G.nodes()).
    width : list of int 
        width[i] = width of the i-th edge in list(G.edges()).
    node_color :
        node_color[i] = border color of the i-th node in list(G.nodes()).
    edgecolors :
        edgecolors[i] = fill color of the i-th node in list(G.nodes()).
    edge_color :
        edge_color[i] = color of the i-th edge in list(G.edges()).
    node_shape :
        node_shape[i] = shape of the i-th node in list(G.nodes()).
    ax : AxesSubplot
        The coordinates in the entire figure when the drawn figure appears as a subfigure.
    title : str
        The title of the figure.
    """
    for node in list(G.nodes()):
        pos_node = {}
        pos_node[node] = pos[node]
        nx.draw_networkx_nodes(G, pos=pos_node,
                               nodelist=[node],
                               node_size=[node_size[list(G.nodes()).index(node)]],node_color=node_color,linewidths = linewidths,
                               edgecolors=[edgecolors[list(G.nodes()).index(node)]],
                               node_shape=node_shape[list(G.nodes()).index(node)],ax=ax)
    nx.draw_networkx_edges(G, pos=pos, edge_color=edge_color, width=width,ax=ax)
    if ax != None and title != None:
        ax.set_title(title, fontsize=30,pad=-30)
    