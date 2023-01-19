import numpy as np

def indicator(x, endpts):
    '''
    if x value is inside the bound, return 1. Otherwise, return 0

    Require:
        x, left_bound, right_bound must have the same dimension

    Parameters: 

            x: 1 x n vector representing the index of point to check (Time dimension should be excluded)

            endpts: 2d (n x 2) array of index. First dimension is all the spatial dimensions, and second dimension are 
                    left and right bound of the subdomain in terms of index

    `return: 
            1 or 0, should be clear enough
    
    '''
#     if len(x) != len(len(endpts[:, 0])):
#         raise ValueError("Parameter dimensions do not agree.")

    if hasattr(x, "__len__"):
        for i in np.arange(len(x)):
            if x[i] < endpts[i][0] or x[i] > endpts[i][1]:
                return 0
        return 1
    else:
        if x >= endpts[0] and x <= endpts[1]:
            return 1
        return 0

def compute_integral(X, spatiotemporal_grid, t, j, endpts):
    '''
    Parameters: 
    
        X: data grid
        
        spatiotemporal_grid: The spatiotemporal_grid that contains information about spatial and time data points.
        
        j: feature index
        
        endpts: n x 2 array 
            the first column is the left endpoints of the subdomain's each of the n dimensions in terms of index,
            second column is right endpoint of each of the subdomain's each of the n dimensions in terms of index
            
    return:
        nd integral within a subdomain
    '''  
    
#     Since all the spatiotemporal_grid contains indication, time and spatial dimensions, and there must be 1 time dimension
#     the number of spatial is then given as following
    grid_ndim = len(np.shape(spatiotemporal_grid))-2
    
# find weights
#     All the 1D weights will be stored in a 2D matrix as cols
#     sudo_var1: max number of pts per dim.
    weights = []
    for i in np.arange(grid_ndim):
#         +2 to account for the time and indication dimension
        index = [0]*(grid_ndim+2)
        index[i] = slice(None)
        index[-1] = i
#         Time is always the second to last dimension, which is filtered here
        index[-2] = t
        
#         we now get the 1D grid by filtering by the index created
        this_dim = spatiotemporal_grid[index]
        
        weight = get_1D_weight(this_dim, endpts[i])
        weights.append(weight)
    
    W_F = get_full_weight(weights)
    
# We now construct F, the spatial grid within a subdomain
    X_F = retrieve_data_mat(spatiotemporal_grid, X)
    F = filterX(X, j, endpts, t)

    return np.sum(np.multiply(W_F, F))

# Matrix to obtain weight
def get_1D_weight(grid, endpt):
    '''
    Parameters: 
        grid: an 1D array that contains the value of the corresponding dimension of each grid points.
        
        endpts: 1 x 2 array 
            the first element is the left endpoints of this dimensions in terms of index,
            second element is the left endpoints of this dimensions in terms of index,
    '''
    
    if endpt[0] >= endpt[1]:
        raise ValueError("Illegal Endpoints.")
    
#     initialize a bunch of 0,
    weight = np.zeros(endpt[1]-endpt[0])

#     find the index at which we enter Omega_k in this axis
    start = endpt[0]
    end = endpt[1]

#     start and end index has different equation for weight, so we do those first
    weight[0] = 1/2*(grid[start+1]-grid[start])
    weight[-1] = 1/2*(grid[end]-grid[end-1])
    weight[1:-1] = np.array([0.5 * (grid[(start+2):(end)] - grid[start:(end-2)])])
    
    return weight

def get_full_weight(weights):
    '''
    weights: a list of lists, where each inner list is the 1D weight in a dimension. 
    '''
    ndim = len(weights)
    W_F = np.array(weights[0])
    for w in np.arange(ndim-1)+1:
        index = [slice(None)]*(w+1)
        index[-1] = np.newaxis
        W_F = W_F[index] * np.array(weights[w])
        
    return W_F

# Methods to filter data matrix X
def retrieve_data_mat(spatiotemporal_grid, X):
    overallShape = list(np.shape(spatiotemporal_grid)[:-1]) + [np.shape(X)[-1]]
    return X.reshape(overallShape)

def filterX(X, j, bound, t_ind):
#     filter by feature j first
    index = [0]*len(np.shape(X))
    for i in range(np.shape(bound)[0]):
        index[i] = slice(bound[i][0], bound[i][1])
    index[-2] = t_ind
    index[-1] = j
    return X[tuple(index)]

# need to figure out kprime and how to store the endpoints for this entire function
def get_theta_nonloc(X, spatiotemporal_grid, j, k, kprime, endpts):
    '''
    Parameters:
        spatiotemporal_grid: The spatiotemporal_grid that contains information about spatial and time data points.
        j: the index of u that we are looking for
        k: the index of subdomain to be used by the indicator function
        kprime: the index of the subdomain to be used as boundary of integral
        endpts: boundary of each subdomain correspond to each dimension in terms of indexing. 
        
    return: 
        vector Theta^nonloc_p
    '''
#     get how many time points are there
    num_t = np.shape(spatiotemporal_grid)[-2]
#     get how many spatial points are there
    num_x = np.prod(np.shape(spatiotemporal_grid)[:-2])
    
    theta_nonloc_p = np.zeros(num_t*num_x)
    
    for i in np.arange(len(theta_nonloc_p)):
    
        this_t = i % num_t
        this_x = i//num_t

        print(this_t)
        print(this_x)

        coefficient = indicator(this_x, endpts[k])

        integral = compute_integral(X, spatiotemporal_grid, this_t, j, endpts)

        theta_nonloc_p[i] = coefficient * integral
    
    return theta_nonloc_p