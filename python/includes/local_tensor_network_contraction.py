import numpy as np

ALLOW_ACSII = list(range(65, 90)) + list(range(97, 122))
LETTES = [chr(ALLOW_ACSII[i]) for i in range(len(ALLOW_ACSII))]


def einsum_eq_convert(ixs, iy):
    """Generate a einqum eq according to ixs (bonds of contraction tensors) and iy (bonds of resulting tensors)
    
    Parameters
    ----------
    ixs : list of list 
        The list of bonds of contraction tensors, ixs[i][i_k] = the node_id of the i_k-th bond of the i-th tensor in the contraction sequence.
    iy: list
        The list of the corresponding node_ids of open_bonds.
    
    Returns
    -------
    einsum_eq :  str
        The corresponding einsum equation of the contraction, for example, 'ab,b->a' corresponds to the multiplication of a matrix and a vector.
    """
    uniquelabels = list(set(sum(ixs, start=[]) + iy))
    labelmap = {l:LETTES[i] for i, l in enumerate(uniquelabels)}
    einsum_eq = ",".join(["".join([labelmap[l] for l in ix]) for ix in ixs]) + \
          "->" + "".join([labelmap[l] for l in iy])
    return einsum_eq



def local_contraction(G_local,J,h,cavity,open_bond,beta):
    """Contract the local tensor network defined on G_local into a vector with the open_bond.
    
    Parameters
    ----------
    G_local : nx.Graph
        The corresponding subgraph of the local tensor network to be contracted.
    J : array
        The coupling constants array with a shape of [n, n], J[i][j] = J_ij for every edge (i, j) of G and in all other positions of J are filled with 0.
    h : array
        The field array with a shape of [n] and h[i] = h_i.
    cavity : array
        The message vectors array with a shape of [n, n, 2], 
        cavity[a][i] = m_{a → i} when a is a boundary node of G_N_i 
                 and = np.exp(beta * h_a * np.array([1, -1])) otherwise.
    open_bond : int
        The node_id of the open bond in the local tensor network.
    beta : float
        The inverse temperature beta.

    Returns
    -------
    result_vector :  array
        Our algorithm only needs to calculate the case where the local tensor network contains only one open bond, so the result is always a vector.
    """
    ixs = [list(edge) for edge in list(G_local.edges())] + \
          [[bond] for bond in list(G_local.nodes())]
    iy = [open_bond]
    eq_convert = einsum_eq_convert(ixs, iy)
    tensors = []
    for edge in list(G_local.edges()):
        tensor = np.exp(J[edge[0]][edge[1]] * beta * np.array([[1, -1], [-1, 1]]))
        tensor = tensor/np.linalg.norm(tensor)
        tensors.append(tensor)
    for bond in list(G_local.nodes()):
        if bond == open_bond:
            tensor = np.exp(beta * h[bond] * np.array([1, -1]))
            tensor = tensor/np.linalg.norm(tensor)
            tensors.append(tensor)
        else:
            tensor = cavity[bond][open_bond]
            tensor = tensor/np.linalg.norm(tensor)
            tensors.append(tensor)
    z = np.einsum(eq_convert, *tensors, optimize=True)
    result_vector = z / z.sum()
    return result_vector

