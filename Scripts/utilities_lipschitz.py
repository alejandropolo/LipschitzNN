############## LIBRERIAS ##############
import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, Point
from matplotlib.patches import Circle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import ConvexHull
from scipy.linalg import norm


############## FUNCIONES ##############

def jacobian_bounding(D_bounded,W,norm_type=1):
    """Calculate the Jacobian bound using a recursive formula.

    This function computes the Jacobian bound for a given set of parameters and a specified norm type.
    
    Args:
        D_bounded (list): A list of D_k values, where D_k represents the bounded jacobian of the k layer with respecto to the layer k.
        W (list): A list of weights W_k matrices, where W_k represents the weight matrix of the layer k.
        norm_type (int, optional): The norm type to be used in the calculation (default is 1).

    Returns:
        float: The calculated Jacobian bound.

    The recursive formula for the Jacobian bound is as follows:
    J^l_p = J_{l-1}_p * W^l * J_l_l

    Where:
    - J^l_p is the Jacobian term at layer l for input layer p.
    - J_{l-1}_p is the Jacobian term at the previous layer (l-1).
    - W^l is the weight matrix at layer l.
    - J_l_l is the Jacobian term at the same layer l.

    The function iterates over the given D_bounded and W lists to compute the Jacobian bound using the specified norm type.

    """
    output = D_bounded[0]*norm(W[1][1:],ord=norm_type)*D_bounded[1]
    for i in range(len(W)-2):
        output = output*norm(W[i+2][1:],ord=norm_type)*D_bounded[i+2]
    return output

def generate_activations_bounds(D,W,activation,norm_type=1):
    """Generate bounds for activations based on the chosen activation function.

    This function generates bounds for activations based on the chosen activation function and a specified norm type.

    Args:
        D (list):  A list of D_k values, where D_k represents the jacobian of the k layer with respecto to the layer k.
        activation (str): The chosen activation function ('sigmoid' or 'relu').
        norm_type (int, optional): The norm type to be used in the calculation (default is 1).

    Returns:
        list: A list of D_bounded values, representing bounds for activations for each layer.

    The function computes bounds for activations at each layer based on the chosen activation function and norm type.

    """
    D_bounded = []
    ### Dado que la primera función de activación es la identidad se hace un bound por un 1
    D_bounded.append(1)
    for i in range(len(W)-2):
        if activation == 'sigmoid':
            matrix = np.ones(D[i+1][0].shape)*0.25 ## Se podría afinar en el caso de la sigmoidal
        elif activation == 'relu':
            matrix = np.ones(D[i+1][0].shape)*1
        bound = norm(matrix,ord=norm_type)
        D_bounded.append(bound)
    D_bounded.append(1)
    return D_bounded


#### BOUNDING THE HESSIAN USING THE NORM OF THE GRADIENTS AND THE WEIGHTS OF THE NETWORK ####
def hessian_bounding(H_bounded,D_bounded,W,norm_type=1):
    """Calculate the Hessian bound using a recursive formula.

    This function computes the Hessian bound for a given set of parameters and a specified norm type.

    Args:
        H_bounded (list): A list of H_k values, where H_k represents the bounded hessian of the layer k wrt to input k.
        D_bounded (list): A list of D_k values, where D_k represents the bounded jacobian of the k layer with respecto to the layer k.
        W (list): A list of W_k matrices, where W_k represents the weight matrix.
        norm_type (int, optional): The norm type to be used in the calculation (default is 1).

    Returns:
        float: The calculated Hessian bound.

    The recursive formula for the Hessian bound is as follows:
    H^l_p = (J_{l-1}_p * W^l) * (J_{l-1}_p * W^l) * H_l_l + H_{l-1}_p * W^l * J_l_l

    Where:
    - H^l_p is the Hessian term at layer l wrt to layer p.
    - J_{l-1}_p is the Jacobian term at the previous layer (l-1) wrt to layer p.
    - W^l is the weight matrix at layer l.
    - J_l_l is the Jacobian term at the same layer l.
    - D_bounded[k], W[k], and H_bounded[k] represent terms for the Jacobian, weight matrix, and Hessian at layer k.

    The function starts with the first term in the recursive formula and iterates over the given D_bounded and W lists to compute the Hessian bound using the specified norm type.

    """

    result = (D_bounded[0]*norm(W[1][1:],ord=norm_type)**2)*H_bounded[1] + H_bounded[0]*norm(W[1][1:],ord=norm_type)*D_bounded[1]
    for i in range(len(W)-2):
        jac = jacobian_bounding(D_bounded[:i+2],W[:i+2],norm_type=norm_type) 
        result = (jac*norm(W[i+2][1:],ord=norm_type)**2)*H_bounded[i+2] + result*norm(W[i+2][1:],ord=norm_type)*D_bounded[i+2]
    return result




def generate_hessian_bounds(H,W,activation,norm_type=1):
    H_bounded = []
    ### Dado que la primera función de activación es la identidad se hace un bound por un 0
    H_bounded.append(0)
    for i in range(len(W)-2):
        if activation == 'sigmoid':
            matrix = np.ones(H[i+1][0][0].shape)*0.25
        elif activation == 'relu':     
            matrix = np.ones(H[i+1][0][0].shape)*0
        else:
            matrix = np.ones(H[i+1][0][0].shape)*1
        bound = norm(matrix,ord=norm_type)
        H_bounded.append(bound)
    H_bounded.append(1)
    return H_bounded