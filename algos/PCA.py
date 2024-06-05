import numpy as np
"""
GOAL: Find a new set of dimensions s.t. all the dimensions are orthogonal and ranked according to the variance of data along them
1. Transformed features are linearly independent
2. Dimensionality can be reduced by taking only the dimensions w/ the highest importance
3. Newly found dimensions should minimize the projection error
4. Projected points should have a maximal spread of points along principal axis(max variance)

- Variance: Var(X) = (1/n) * sum[(X_i - X_bar)^2]
    - average square distance from the mean
    
- Projections:
    - orthogonal: projects a vector onto a subspace in such a way that the diff b/w the vector and its projection is orthogonal(perpendicular) to the subspace
    
- Covariance matrix:
    Cov(X, Y) = (1/n) * sum[(X_i - X_bar)(Y_i - Y_bar).T]
    Cov(X, X) = (1/n) * sum[(X_i - X_bar)(X_i - X_bar).T]
    
- Eigenvalues: 
    - Represent amount of variance explained by each principal component
    - Find eigenvalues that maximizes variance (principal component)
- Eigenvectors: Represent the direction of the principal component

APPROACH:
1. Subtract mean from X
2. Calculate Cov(X,X)
3. Calculate eidenvectors and eigenvalues of cov matrix
4. Sort eigen vectors by their eigenvalues in decr order
5. choose first k eigenvectors and that will be new k dimensions
6. Transform og dimensional data poins into k dimensions (projection w/ dot prod.)
"""
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
    
    def fit(self, X):
        # mean
        self.mean = np.mean(X)
        X = X - self.mean
        
        # covariance
        # have: row -> sample, col -> feature
        # need: col -> sample, row -> feature
        cov = np.cov(X.T)
        
        # eigenvectors, eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # sort eigenvectors decr and get indices
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        
        # store first n eigenvectors 
        self.components = eigenvectors[:self.n_components]
        
    
    def transform(self, X):
        # project X
        X = X - self.mean
        return np.dot(X, self.components.T)
    
# Testing
if __name__ == "__main__":
    # Imports
    import matplotlib.pyplot as plt
    from sklearn import datasets

    # data = datasets.load_digits()
    data = datasets.load_iris()
    X = data.data
    y = data.target

    # Project the data onto the 2 primary principal components
    pca = PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print("Shape of X:", X.shape)
    print("Shape of transformed X:", X_projected.shape)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    plt.scatter(
        x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    plt.show()