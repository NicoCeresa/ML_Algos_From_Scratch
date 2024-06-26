import numpy as np
"""
GOAL: Find a hyperplane that maximizes the distance between support vector
"""

class SVM:
    
    def __init__(self, learning_rate=1e-3, lambda_param=1e-2, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.lr * y_[idx]
    
    
    def predict(self, X):
        approx = np.dot(X, self.weights) - self.b
        return np.sign(approx)
    
    
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn import datasets
    import matplotlib.pyplot as plt
    
    X, y = datasets.make_blobs(
        n_samples = 50,
        n_features = 2,
        centers = 2,
        cluster_std = 1.05,
        random_state = 42
    )
    # make sure classes -1 and 1
    y = np.where(y == 0, -1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    clf = SVM()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    
    print(f'Accuracy of SVM is {accuracy_score(y_test, preds)}')
    
    def visualize_svm():
        def get_hyperplane(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plt.scatter(X[:,0], X[:,1], marker='o', c=y)

        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])
        
        x1_1 = get_hyperplane(x0_1, clf.w, clf.b, 0)
        x1_2 = get_hyperplane(x0_2, clf.w, clf.b, 0)
        
        x1_1_m = get_hyperplane(x0_1, clf.w, clf.b, -1)
        x1_2_m = get_hyperplane(x0_2, clf.w, clf.b, -1)
        
        x1_1_p = get_hyperplane(x0_1, clf.w, clf.b, 1)
        x1_2_p = get_hyperplane(x0_2, clf.w, clf.b, 1)
        
        ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")
        
        x1_min = np.amin(X[:, 1])
        x1_max = np.amax(X[:, 1])
        ax.set_ylim([x1_min-3, x1_max+3])
        
        plt.show()
        
    visualize_svm()