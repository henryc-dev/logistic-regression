# Logistic Regression Model from Scratch
Build a simple logistic regression model from scratch and test it with a binary classification class. The key idea is to generalize the linear regression result to a classification model by finding a <b>link function</b> $g$ that bridge linear regression to probabilistic values.

$$ g(P(y=i)) = \sum_{i=0}^{n}\beta_ix_i $$

One way to do so is to find a smooth and well-shaped function that maps $(-\infty, \infty)$ to (0, 1), which we call the <b>signmoid functon</b>:

$$S(x) = \frac{1}{1+e^{-y}} = p$$


$\text{where } y = \sum\limits_{i=0}^{n}\beta_ix_i\$

## Data
Iris dataset: https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html

## Results
Achieved an accuracy of 1.0
