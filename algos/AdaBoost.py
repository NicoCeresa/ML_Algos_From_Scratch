import numpy as np
"""
AdaBoost

- uses boosting approach
- BOOSTING: combines multiple weak classifiers into one powerful one

Example
- have samples with 2 diff features
- first classifier makes split based on y axis
- some good preds but some missed classifications
- use accuracy to calculate and update weights
- uses weights to find new decision boundary
- repeat this
- at end combine classifiers to draw good decision line
    - more complex than linear decision line
    
Weak Learner (decision stump: decision tree with only 1 split): simple classifier
- if > or < than threshold then -1 or +1
- Error: # missclassifications/# samples in first iter
    - after first iter: sum of missclassified weights
    - if error > 0.5 flip decision and the error = 1 - error
- Weights: w_0 = 1/N
    - w = [w * exp(-a * y * h(X))]/sum(w) ; where h(X) = pred of t
- Performance: a = 0.5 * log((1-e)/e)
- Prediction: y = sign(sum(a_t * h(X)))
- Training: 
    - for t in T:
        - train weak classifier
        - calc error weights
        - calc a
        - update weights
"""
class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None
        
    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        
        preds = np.ones(n_samples)
        if self.polarity == 1:
            preds[X_column < self.threshold] = -1
        else:
            preds[X_column > self.threshold] = -1
        return preds
        
        
class Adaboost:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        # init weights
        w = np.full(n_samples, (1/n_samples))
        # iter through clfs
        self.clfs = []
        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float('inf')
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)
                for threshold in thresholds:
                    p = 1
                    preds = np.ones(n_samples)
                    preds[X_column < threshold] = -1
                    
                    missclassified = w[y != preds]
                    error = np.sum(missclassified)
                    
                    if error > 0.5:
                        error = 1-error
                        p = -1
                        
                    if error < min_error:
                        min_error = error
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i
                        
            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1-error)/(error+EPS))
            preds = clf.predict(X)

            w *= np.exp(-clf.alpha * y * preds)
            # normalize
            w /= np.sum(w)
            
            self.clfs.append(clf)
    
    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred) 
        return y_pred
    
# Testing
if __name__ == "__main__":
    # Imports
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    y[y == 0] = -1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=5
    )

    # Adaboost classification with 5 weak classifiers
    clf = Adaboost(n_clf=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy(y_test, y_pred)
    print("Accuracy:", acc)