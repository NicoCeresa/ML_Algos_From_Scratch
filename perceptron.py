"""
Approximation:
y_hat= g(f(w,b)) = g(w.T * x + b)

Update rule:
w := w + delta(w)
delta(w) := alpha * (y_i - y_hat_i) * x_i

alpha: learning rate

Explanation:
- if same label the no change, else change
- if misclassification push weights in + or - direction an alpha amount
y  y_hat  y-y_hat
1    1      0
1    0      1
0    0      0
0    1     -1
"""
import numpy as np
class Perceptron:
    """
    Implimentation of the perceptron algorithm using only NumPy
    """
    def __init__(self, learning_rate=1e-2, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = np.array([1 if i > 0 else 0 for i in y])
        
        for _ in range(self.n_iters):
            for idx, X_i in enumerate(X):
                linear_out = np.dot(X_i, self.weights) + self.bias
                y_pred = self.activation_func(linear_out)
                delta = y_[idx] - y_pred
                self.weights += self.lr * delta * X_i
                self.bias += self.lr * delta
    
    def predict(self, x):
        linear_out = np.dot(x, self.weights) + self.bias
        return self.activation_func(linear_out)
        
    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, 0)
    
# Testing
if __name__ == "__main__":
    # Imports
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    from perceptron import Perceptron

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    X, y = datasets.make_blobs(
        n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    p = Perceptron(learning_rate=0.01, n_iters=1000)
    p.fit(X_train, y_train)
    predictions = p.predict(X_test)

    print("Perceptron classification accuracy", accuracy(y_test, predictions))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    ymin = np.amin(X_train[:, 1])
    ymax = np.amax(X_train[:, 1])
    ax.set_ylim([ymin - 3, ymax + 3])

    plt.show()