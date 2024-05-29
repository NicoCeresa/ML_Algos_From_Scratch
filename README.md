# Implimentations of Machine Learning Algos using only NumPy

## Algos So Far:
- Linear Regression
- Logistic Regression
- Naive Bayes
- K-Nearest Neighbors
- SVM
- Perceptron
## KNN
- GOAL: Classify a point based on the majority label of it's K nearest neighbors
- KEY: K must be odd as we cannot classify if there are an equal amount of each class

### Steps
1. Calculate the Euclidean Distance using this formula: <br/>
$d(p,q) = \sqrt{\sum_{i=1}^n(q_i - p_i)^2}$

2. Once you have calculated the distances between your point and all other training points, keep only the nearest K neighbors

3. Find the label of those neighbors

## SVM
- **GOAL:** find a decision boundary that maximizes the distance between the plane and nearest data points
- **KEY:** These nearest points are the SUPPORT VECTORS

Lets say we have decision boundaries given by the formulas:
1. $w * x_+ - b \geq 1$ if $y_i == 1$ <br/>
2. $w * x_- - b \leq -1$ if $y_i == -1$

The distance between these decision boundaries can be expressed as: <br/>
Distance = $\frac{2}{||w||}$

$y_i(w * x_i - b) \geq 1$ for $y_i \in \{-1, 1\}$

Using our knowledge that $y_i == 1$ or $y_i == -1$ we multiply both equations by the label: $y_i$ and derive the following:

1. $y_i(w * x_+ - b) \geq 1 * y_i = 1$ <br/>
2. $y_i(w * x_- - b) \leq -1 * y_i = -1$

In equation 2 we flip the sign because we are multiplying by a negative

1. $y_i(w * x_+ - b) \geq 1$ <br/>
2. $y_i(w * x_- - b) \geq 1$

Now, we just have one singular equation:<br/>
$y_i(w * x_i - b) \geq 1$

### Loss Function: Hinge Loss
- The further the data point is on the wrong side, the higher the loss

$loss = max(0, 1 - y_i(w * x_i - b))$

Or, in other words: <br/>
let $f(x) = w * x_i - b$<br/>
$loss = 0$ if $y * f(x) >= 1$ else $loss = 1 - y * f(x)$

### Regularization
- trade off between minimizing loss and maximizing the distance to both sides

$J = \lambda||w||^2 + \frac{1}{n}\sum{max(0,1 - y_i(w * x_i - b))}$

if $y_i * f(x) \geq 1:$<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $J_i = \lambda||w||^2$

else:<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $J_i = \lambda||w||^2 + 1 - y_i(w * x_i - b)$

### Gradients/Derivatives

if $y_i * f(x) \geq 1:$<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\frac{dJ_i}{dw_k} = 2\lambda w_k$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\frac{dJ_i}{db} = 0$

else:<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\frac{dJ_i}{dw_k} = 2\lambda w_k - y_i * x_{ik}$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\frac{dJ_i}{db} = y_i$

### Update Rule

if $y_i * f(x) \geq 1:$<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $w = w - \alpha * dw = w - \alpha * 2\lambda w$<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $b = b - \alpha * db = b$

else:<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $w = w - \alpha * dw = w - \alpha * (2\lambda w - y_i * x_i)$<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $b = b - \alpha * db = b - \alpha * y_i$

### Steps To Code
1. Training<br/>
    a. Init Weights <br/>
    b. Make sure $y \in \{-1, 1\}$
    c. Apply update rules for n_iters

2. Prediction<br/>
    a. Calculate $y = sign(w * x - b)$