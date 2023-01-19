import warnings
from itertools import combinations
from itertools import combinations_with_replacement as combinations_w_r
from itertools import product as iproduct

import numpy as np
from sklearn import __version__
from sklearn.utils.validation import check_is_fitted

from ..utils import AxesArray
from ..utils import comprehend_axes
from .base import BaseFeatureLibrary
from .base import x_sequence_or_item
from pysindy.differentiation import FiniteDifference


class PDELibrary(BaseFeatureLibrary):
    """Generate a PDE library with custom functions.

    Parameters
    ----------
    library_functions : list of mathematical functions, optional (default None)
        Functions to include in the library. Each function will be
        applied to each input variable (but not their derivatives)

    derivative_order : int, optional (default 0)
        Order of derivative to take on each input variable,
        can be arbitrary non-negative integer.

    spatial_grid : np.ndarray, optional (default None)
        The spatial grid for computing derivatives

    temporal_grid : np.ndarray, optional (default None)
        The temporal grid if using SINDy-PI with PDEs.

    function_names : list of functions, optional (default None)
        List of functions used to generate feature names for each library
        function. Each name function must take a string input (representing
        a variable name), and output a string depiction of the respective
        mathematical function applied to that variable. For example, if the
        first library function is sine, the name function might return
        :math:`\\sin(x)` given :math:`x` as input. The function_names list
        must be the same length as library_functions.
        If no list of function names is provided, defaults to using
        :math:`[ f_0(x),f_1(x), f_2(x), \\ldots ]`.

    interaction_only : boolean, optional (default True)
        Whether to omit self-interaction terms.
        If True, function evaulations of the form :math:`f(x,x)` and
        :math:`f(x,y,x)` will be omitted, but those of the form :math:`f(x,y)`
        and :math:`f(x,y,z)` will be included.
        If False, all combinations will be included.

    include_bias : boolean, optional (default False)
        If True (default), then include a bias column, the feature in which
        all polynomial powers are zero (i.e. a column of ones - acts as an
        intercept term in a linear model).
        This is hard to do with just lambda functions, because if the system
        is not 1D, lambdas will generate duplicates.

    include_interaction : boolean, optional (default True)
        This is a different than the use for the PolynomialLibrary. If true,
        it generates all the mixed derivative terms. If false, the library
        will consist of only pure no-derivative terms and pure derivative
        terms, with no mixed terms.

    library_ensemble : boolean, optional (default False)
        Whether or not to use library bagging (regress on subset of the
        candidate terms in the library)

    ensemble_indices : integer array, optional (default [0])
        The indices to use for ensembling the library.

    implicit_terms : boolean
        Flag to indicate if SINDy-PI (temporal derivatives) is being used
        for the right-hand side of the SINDy fit.

    multiindices : list of integer arrays,  (default None)
        Overrides the derivative_order to customize the included derivative
        orders. Each integer array indicates the order of differentiation
        along the corresponding axis for each derivative term.

    differentiation_method : callable,  (default FiniteDifference)
        Spatial differentiation method.

     diff_kwargs: dictionary,  (default {})
        Keyword options to supply to differtiantion_method.

    Attributes
    ----------
    functions : list of functions
        Mathematical library functions to be applied to each input feature.

    function_names : list of functions
        Functions for generating string representations of each library
        function.

    n_input_features_ : int
        The total number of input features.
        WARNING: This is deprecated in scikit-learn version 1.0 and higher so
        we check the sklearn.__version__ and switch to n_features_in if needed.

    n_output_features_ : int
        The total number of output features. The number of output features
        is the product of the number of library functions and the number of
        input features.

    Examples
    --------
    >>> import numpy as np
    >>> from pysindy.feature_library import PDELibrary
    """

    def __init__(
        self,
        library_functions=[],
        derivative_order=0,
        spatial_grid=None,
        temporal_grid=None,
        interaction_only=True,
        function_names=None,
        include_bias=False,
        include_interaction=True,
        library_ensemble=False,
        ensemble_indices=[0],
        implicit_terms=False,
        multiindices=None,
        differentiation_method=FiniteDifference,
        diff_kwargs={},
        is_uniform=None,
        periodic=None,
    ):
        super(PDELibrary, self).__init__(
            library_ensemble=library_ensemble, ensemble_indices=ensemble_indices
        )
        self.functions = library_functions
        self.derivative_order = derivative_order
        self.function_names = function_names
        self.interaction_only = interaction_only
        self.implicit_terms = implicit_terms
        self.include_bias = include_bias
        self.include_interaction = include_interaction
        self.num_trajectories = 1
        self.differentiation_method = differentiation_method
        self.diff_kwargs = diff_kwargs

        if function_names and (len(library_functions) != len(function_names)):
            raise ValueError(
                "library_functions and function_names must have the same"
                " number of elements"
            )
        if derivative_order < 0:
            raise ValueError("The derivative order must be >0")

        if is_uniform is not None or periodic is not None:
            # DeprecationWarning are ignored by default...
            warnings.warn(
                "is_uniform and periodic have been deprecated."
                "in favor of differetiation_method and diff_kwargs.",
                UserWarning,
            )

        if (spatial_grid is not None and derivative_order == 0) or (
            spatial_grid is None and derivative_order != 0 and temporal_grid is None
        ):
            raise ValueError(
                "Spatial grid and the derivative order must be "
                "defined at the same time if temporal_grid is not being used."
            )

        if temporal_grid is None and implicit_terms:
            raise ValueError(
                "temporal_grid parameter must be specified if implicit_terms "
                " = True (i.e. if you are using SINDy-PI for PDEs)."
            )
        elif not implicit_terms and temporal_grid is not None:
            raise ValueError(
                "temporal_grid parameter is specified only if implicit_terms "
                " = True (i.e. if you are using SINDy-PI for PDEs)."
            )
        if spatial_grid is not None and spatial_grid.ndim == 1:
            spatial_grid = spatial_grid[:, np.newaxis]

        if temporal_grid is not None and temporal_grid.ndim != 1:
            raise ValueError("temporal_grid parameter must be 1D numpy array.")
        if temporal_grid is not None or spatial_grid is not None:
            if spatial_grid is None:
                spatiotemporal_grid = temporal_grid
                spatial_grid = np.array([])
            elif temporal_grid is None:
                spatiotemporal_grid = spatial_grid[
                    ..., np.newaxis, :
                ]  # append a fake time axis
            else:
                spatiotemporal_grid = np.zeros(
                    (
                        *spatial_grid.shape[:-1],
                        len(temporal_grid),
                        spatial_grid.shape[-1] + 1,
                    )
                )
                for ax in range(spatial_grid.ndim - 1):
                    spatiotemporal_grid[..., ax] = spatial_grid[..., ax][
                        ..., np.newaxis
                    ]
                spatiotemporal_grid[..., -1] = temporal_grid
        else:
            spatiotemporal_grid = np.array([])
            spatial_grid = np.array([])

        self.spatial_grid = spatial_grid

        # list of derivatives
        indices = ()
        if np.array(spatiotemporal_grid).ndim == 1:
            spatiotemporal_grid = np.reshape(
                spatiotemporal_grid, (len(spatiotemporal_grid), 1)
            )

        # if want to include temporal terms -> range(len(dims))
        if self.implicit_terms:
            self.ind_range = spatiotemporal_grid.ndim - 1
        else:
            self.ind_range = spatiotemporal_grid.ndim - 2

        for i in range(self.ind_range):
            indices = indices + (range(derivative_order + 1),)

        if multiindices is None:
            multiindices = []
            for ind in iproduct(*indices):
                current = np.array(ind)
                if np.sum(ind) > 0 and np.sum(ind) <= self.derivative_order:
                    multiindices.append(current)
            multiindices = np.array(multiindices)
        num_derivatives = len(multiindices)

        self.num_derivatives = num_derivatives
        self.multiindices = multiindices
        self.spatiotemporal_grid = AxesArray(
            spatiotemporal_grid, comprehend_axes(spatiotemporal_grid)
        )

    @staticmethod
    def _combinations(n_features, n_args, interaction_only):
        """Get the combinations of features to be passed to a library function."""
        comb = combinations if interaction_only else combinations_w_r
        return comb(range(n_features), n_args)

    def get_feature_names(self, input_features=None):
        """Return feature names for output features.

        Parameters
        ----------
        input_features : list of string, length n_features, optional
            String names for input features if available. By default,
            "x0", "x1", ... "xn_features" is used.

        Returns
        -------
        output_feature_names : list of string, length n_output_features
        """
        check_is_fitted(self)
        if float(__version__[:3]) >= 1.0:
            n_features = self.n_features_in_
        else:
            n_features = self.n_input_features_

        if input_features is None:
            input_features = ["x%d" % i for i in range(n_features)]
        if self.function_names is None:
            self.function_names = list(
                map(
                    lambda i: (lambda *x: "f" + str(i) + "(" + ",".join(x) + ")"),
                    range(n_features),
                )
            )
        feature_names = []

        # Include constant term
        if self.include_bias:
            feature_names.append("1")

        # Include any non-derivative terms
        for i, f in enumerate(self.functions):
            for c in self._combinations(
                n_features, f.__code__.co_argcount, self.interaction_only
            ):
                feature_names.append(
                    self.function_names[i](*[input_features[j] for j in c])
                )

        def derivative_string(multiindex):
            ret = ""
            for axis in range(self.ind_range):
                if self.implicit_terms and (
                    axis
                    in [
                        self.spatiotemporal_grid.ax_time,
                        self.spatiotemporal_grid.ax_sample,
                    ]
                ):
                    str_deriv = "t"
                else:
                    str_deriv = str(axis + 1)
                for i in range(multiindex[axis]):
                    ret = ret + str_deriv
            return ret

        # Include derivative terms
        for k in range(self.num_derivatives):
            for j in range(n_features):
                feature_names.append(
                    input_features[j] + "_" + derivative_string(self.multiindices[k])
                )
        # Include mixed non-derivative + derivative terms
        if self.include_interaction:
            for k in range(self.num_derivatives):
                for i, f in enumerate(self.functions):
                    for c in self._combinations(
                        n_features,
                        f.__code__.co_argcount,
                        self.interaction_only,
                    ):
                        for jj in range(n_features):
                            feature_names.append(
                                self.function_names[i](*[input_features[j] for j in c])
                                + input_features[jj]
                                + "_"
                                + derivative_string(self.multiindices[k])
                            )
        return feature_names

    @x_sequence_or_item
    def fit(self, x_full, y=None):
        """Compute number of output features.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Measurement data.

        Returns
        -------
        self : instance
        """
        n_features = x_full[0].shape[x_full[0].ax_coord]

        if float(__version__[:3]) >= 1.0:
            self.n_features_in_ = n_features
        else:
            self.n_input_features_ = n_features

        n_output_features = 0
        # Count the number of non-derivative terms
        n_output_features = 0
        for f in self.functions:
            n_args = f.__code__.co_argcount
            n_output_features += len(
                list(self._combinations(n_features, n_args, self.interaction_only))
            )

        # Add the mixed derivative library_terms
        if self.include_interaction:
            n_output_features += n_output_features * n_features * self.num_derivatives
        # Add the pure derivative library terms
        n_output_features += n_features * self.num_derivatives

        # If there is a constant term, add 1 to n_output_features
        if self.include_bias:
            n_output_features += 1

        self.n_output_features_ = n_output_features

        # required to generate the function names
        self.get_feature_names()

        return self

    @x_sequence_or_item
    def transform(self, x_full):
        """Transform data to pde features

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The data to transform, row by row.

        Returns
        -------
        xp : np.ndarray, shape (n_samples, n_output_features)
            The matrix of features, where n_output_features is the number of
            features generated from the tensor product of the derivative terms
            and the library_functions applied to combinations of the inputs.
        """
        check_is_fitted(self)

        xp_full = []
        for x in x_full:
            n_features = x.shape[x.ax_coord]

            if float(__version__[:3]) >= 1.0:
                if n_features != self.n_features_in_:
                    raise ValueError("x shape does not match training shape")
            else:
                if n_features != self.n_input_features_:
                    raise ValueError("x shape does not match training shape")

            shape = np.array(x.shape)
            shape[-1] = self.n_output_features_
            xp = np.empty(shape, dtype=x.dtype)

            # derivative terms
            shape[-1] = n_features * self.num_derivatives
            library_derivatives = np.empty(shape, dtype=x.dtype)
            library_idx = 0
            for multiindex in self.multiindices:
                derivs = x
                for axis in range(self.ind_range):
                    if multiindex[axis] > 0:
                        s = [0 for dim in self.spatiotemporal_grid.shape]
                        s[axis] = slice(self.spatiotemporal_grid.shape[axis])
                        s[-1] = axis

                        derivs = self.differentiation_method(
                            d=multiindex[axis],
                            axis=axis,
                            **self.diff_kwargs,
                        )._differentiate(derivs, self.spatiotemporal_grid[tuple(s)])
                library_derivatives[
                    ..., library_idx : library_idx + n_features
                ] = derivs
                library_idx += n_features

            # library function terms
            n_library_terms = 0
            for f in self.functions:
                for c in self._combinations(
                    n_features, f.__code__.co_argcount, self.interaction_only
                ):
                    n_library_terms += 1

            shape[-1] = n_library_terms
            library_functions = np.empty(shape, dtype=x.dtype)
            library_idx = 0
            for f in self.functions:
                for c in self._combinations(
                    n_features, f.__code__.co_argcount, self.interaction_only
                ):
                    library_functions[..., library_idx] = f(*[x[..., j] for j in c])
                    library_idx += 1

            library_idx = 0

            # constant term
            if self.include_bias:
                shape[-1] = 1
                xp[..., library_idx] = np.ones(shape[:-1], dtype=x.dtype)
                library_idx += 1

            # library function terms
            xp[..., library_idx : library_idx + n_library_terms] = library_functions
            library_idx += n_library_terms

            # pure derivative terms
            xp[
                ..., library_idx : library_idx + self.num_derivatives * n_features
            ] = library_derivatives
            library_idx += self.num_derivatives * n_features

            # mixed function derivative terms
            shape[-1] = n_library_terms * self.num_derivatives * n_features
            if self.include_interaction:
                xp[
                    ...,
                    library_idx : library_idx
                    + n_library_terms * self.num_derivatives * n_features,
                ] = np.reshape(
                    library_functions[..., np.newaxis, :]
                    * library_derivatives[..., :, np.newaxis],
                    shape,
                )
                library_idx += n_library_terms * self.num_derivatives * n_features
            xp = AxesArray(xp, comprehend_axes(xp))
            xp_full.append(xp)
        if self.library_ensemble:
            xp_full = self._ensemble(xp_full)
            
        print(xp_full.shape)
        return xp_full

    def get_spatial_grid(self):
        return self.spatial_grid
    
    
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#   nonlocal methods start here
    def setup_subdomains(spatial_grid, K):
        """Setup subdomains for nonlocal computation

            Output format: A list of ndim elements containing lists of bounds of subdomains. In a 2D
            case, an example would be [[[0, 1], [2, 4], [5, 9]], [[0, 4], [5, 12]]].
        """
    #       spatial grid is this 2D tensor that stores the grid. 
    #       We split each dimension into K (mostly) equally sized subdomains, where K is a parameter. 
    #       There will be a total of K^ndim subdomains 
        bounds = []
    #     -1 to adjust for the last dimension, which indicates feature.
        for i in np.arange(np.array(spatial_grid).ndim-1):
    #         does spatio-temporal grid have to have the same number of points in each spatial dimension?
            length = np.shape(spatial_grid)[1]

            if length//K[i] < 2:
    #             replace this with warning or break point
                print("Warning: too many subdomains created in axis", i)

            subdomain_length = length//K[i]
            remain = length % K[i]
            size = np.zeros(K[i]) + subdomain_length
            size[:remain] += 1
            bound = np.cumsum(size)
            bound = [int(s) for s in bound]
            bounds.append(bound)
        return bounds

    '''A generalization from https://stackoverflow.com/questions/29142417/4d-position-from-1d-index'''
    def nd_iterator(index, K):
        '''
        index: the 1D index of the nd item to recover
        K: number of items per dimension, corresponding to each of the n dimensions.

        return:
        nd index of the item with 1D index to be "index"
        '''
        nd_index = [index % K[0]]
        dividor = 1
        remaining_index = index
        for i in np.arange(len(K)-1):
            remaining_index -= nd_index[i]*dividor
            dividor *= K[i]
            this_index = remaining_index // dividor % K[i+1]
            nd_index.append(this_index)
        return nd_index

    def subdomain_iterator(index, bounds, K):
        '''
        bounds: subdomain dividor, in standard setup_subdomains format
        nd_index: nd index of a specific subdomain, in nd_iterator standard output format.

        return:
        boundary of that subdomain in standard nonlocal format.
        '''
        nd_index = nd_iterator(index, K)
        bound = []
        for i in np.arange(len(nd_index)):
            if nd_index[i] == 0:
                bound.append([0, bounds[i][0]-1])
            else:
                bound.append([bounds[i][nd_index[i]-1], bounds[i][nd_index[i]]-1])

        return np.reshape(bound, (len(nd_index), 2))

    def spatial_iterator(index, spatiotemporal_grid, K):
        '''
        bounds: subdomain dividor, in standard setup_subdomains format
        nd_index: nd index of a specific subdomain, in nd_iterator standard output format.

        return:
        nd value of x at that index.
        '''
        nd_index = nd_iterator(index, K)
    #     append time point
        nd_index.append(0)

    #     compute slicing index for the last dimension
        num_dim = len(np.shape(spatiotemporal_grid))-1
    #     append last axis
        nd_index.append(slice(0, num_dim-1, 1))

        return spatiotemporal_grid[tuple(nd_index)]

    #indicator function has been modified
    # modify it so it can kicks the points with not enough dimension to be zero
    def indicator(x_points, endpts):
        '''
        if x value is inside the bound, return 1. Otherwise, return 0

        Require:
            endpts: (left_bound, right_bound).
            x, left_bound, right_bound must have the same dimension

        Parameters: 

                x: m x (n+1) vector representing the index of point to check (Time dimension should be excluded)

                endpts: 2d (n x 2) array of index. First dimension is all the spatial dimensions, and second dimension are 
                        left and right bound of the subdomain in terms of index

        `return: 
                1 x n matrix that consist of 1 and 0's indicating whether the point is inside the subdomain or not
        '''
        x = np.copy(x_points)
        x = x.T

        # iterate through each dimensions
        for i in np.arange(np.shape(endpts)[0]):
            upper_limit = endpts[i][1]
            lower_limit = endpts[i][0]
            x[i] = np.multiply(np.abs(x[i] - 0.5*(upper_limit+lower_limit))<=0.5*(upper_limit-lower_limit), 1)

        # Transpose back to original shape
        x = x.T
        return np.add(np.sign(np.subtract(np.sum(x, axis=1), len(endpts))), 1)

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
            W_F = W_F[tuple(index)] * np.array(weights[w])

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

    def get_theta_nonloc(X, spatiotemporal_grid, j, k, kprime, bounds, K):
        '''
        Parameters:
            spatiotemporal_grid: The spatiotemporal_grid that contains information about spatial and time points.
            j: the index of u that we are looking for
            k: the index of subdomain to be used by the indicator function
            kprime: the index of the subdomain to be used as boundary of integral
            bounds: boundary of each subdomain correspond to each dimension in terms of indexing. 

        return: 
            vector Theta^nonloc_p
        '''
        spatio_grid = spatiotemporal_grid[..., 0, :-1]
        s = time.time()
    #     get number and space and time points
        num_t = np.shape(spatiotemporal_grid)[-2]
        num_x = np.prod(np.shape(spatiotemporal_grid)[:-2])


    #     Since all the spatiotemporal_grid contains indication, time and spatial dimensions, and there must be 1 time dimension
    #     the number of spatial is then given as following
        grid_ndim = len(np.shape(spatiotemporal_grid))-2

    #     construct shape of x
        x_shape = np.shape(spatiotemporal_grid)[:-2]

        coeff = np.zeros(num_t*num_x)
        integral = np.zeros(num_t*num_x)

    #     x loop
        coeff_bounds = subdomain_iterator(k, bounds, K)
        x_flat = np.zeros((2, 4096))
        x_flat[0, :] = np.reshape(spatio_grid[:, :, 0], 4096)
        x_flat[1, :] = np.reshape(spatio_grid[:, :, 1], 4096)

        out = indicator(x_flat.T, coeff_bounds)
        coeff = np.kron(out, [1]*num_t)

    #    t loop
        s_T = time.time()
        integral_bounds = subdomain_iterator(kprime, bounds, K)
        for t in np.arange(num_t):
            # find weights
            # All the 1D weights will be stored in a 2D matrix as cols
            weights = []
            for i in np.arange(grid_ndim):
                # +2 to account for the time and indication dimension
                index = [0]*(grid_ndim+2)
                # Time is always the second to last dimension, which is filtered here
                index[-2] = t
                index[i] = slice(None)
                index[-1] = i

        #         we now get the 1D grid by filtering by the index created
                this_dim = spatiotemporal_grid[tuple(index)]
                weight = get_1D_weight(this_dim, integral_bounds[i])
                weights.append(weight)

            W_F = get_full_weight(weights)
            F = filterX(X, j, integral_bounds, t)

            integral[t::num_t] = np.sum(np.multiply(W_F, F))
        print("T loop time:", time.time()-s_T)

        s_M = time.time()
        theta_nonloc_p = coeff * integral
        print("multiply time:", time.time()-s_M)

        return theta_nonloc_p
    
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#     Spatial grid is pre_defined. 
    def sample_test_space(spatiotemporal_grid, x):
    
        dims = spatial_grid.shape[:-1]
        grid_ndim = len(dims)

    #     number of space points sampled
        n_samples_full = np.prod(np.shape(x)[:grid_ndim])
    #     number of features at each space point
        n_features = np.shape(x)[-1]-1

        K = [2]*grid_ndim
        subdomain_bounds = setup_subdomains(spatial_grid, K)

        print(subdomain_bounds)

        res = []

        tot_iter = n_features * np.prod(K) * np.prod(K)

        for j in np.arange(n_features):
            for k in np.arange(np.prod(K)):
                for kprime in np.arange(np.prod(K)):

                    start = time.time()
                    res.append(get_theta_nonloc(x, spatiotemporal_grid, j, k, kprime, subdomain_bounds, K))
                    end = time.time()-start
        return res
