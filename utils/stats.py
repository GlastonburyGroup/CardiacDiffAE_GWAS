import numpy as np
import scipy.stats
import scipy.linalg

def estimate_meff(data, method="Li"):
    """
    Estimates the effective number of tests in correlated datasets using two different 
    approaches (Li's is to be preferred).

    Parameters:
    data (numpy.ndarray): Data matrix.
    method (str): Method to be used for estimation. Either "Cheverud" or "Li".

    Returns:
    float: Effective number of tests.
    """
    M = data.shape[1]
    cor = np.corrcoef(data, rowvar=False)
    eigenvalues = scipy.linalg.eigh(cor, eigvals_only=True)

    if method == "Cheverud":
        V_lambda = np.sum((eigenvalues - 1) ** 2) / (M - 1)
        M_eff = 1 + (M - 1) * (1 - (V_lambda / M))
    elif method == "Li":
        eigenvalues = np.abs(eigenvalues)
        M_eff = np.sum(eigenvalues >= 1) + np.sum(eigenvalues - np.floor(eigenvalues))
    else:
        raise ValueError("Method not valid. Valid methods are 'Cheverud' and 'Li'")

    return M_eff


# Example usage:
# data = np.random.normal(size=(10, 20))
# print(estimate_meff(data, "Li"))
