import numpy as np

class NaiveBayes:
    """
    P(y|X) = (P(X|Y) * P(y))/P(X)
    
    X = (x1, x2, ..., xn)
    P(y|X) = (P(x1|y) * P(x2|y) * ... * P(xn|y) * P(y))/p(X)
    p(xi|y) = (exp(-(xi - mu)^2/2*sigam^2) / root(2*pi*sigma^2))
    - gaussian dist ^
        - sigma: variance
        - mu: mean
    
    Select class  w/ Highest Probability
    - as values are probs b\w [0,1], apply log to prevent values from getting too small
    y = argmax[log(P(x1|y)) + log(P(x2|y)) + ... + log(P(xn|Y)) + log(P(y))]
     """
    # no init
     
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        
        # init mean, var, and prior
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)
        
        for c in self._classes:
            X_c = X[c == y]
            self._mean[c, :] = X_c.mean(axis=0)
            self._var[c, :] = X_c.var(axis=0)
            # priors == frequency
            self._priors[c] = X_c.shape[0] / float(n_samples)
        
    
    def predict(self, X):
        y_pred = [self._predict(i) for i in X]
        return y_pred
    
    def _predict(self, x):
        posteriors = []
        
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
            
        return self._classes[np.argmax(posteriors)] 
            
    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-(x - mean)**2 / (2 * var))
        denom = np.sqrt(2 * np.pi * var)
        return numerator / denom
    
# Testing
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=123
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)

    print("Naive Bayes classification accuracy", accuracy(y_test, predictions))