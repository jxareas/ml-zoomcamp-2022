# Week 1: Machine Learning for Regression

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#definition">Definition</a></li>
    <li><a href="#normal-equation">Normal Equation</a></li>
    <li><a href="#categorical-variables">Categorical Variables</a></li>
  </ol>
</details>

## The Linear Regression model

Linear Regression is a very simple approach for Supervised Learning, as it is a
useful tool for predicting quantitative responses.
It serves as a good jumping-off point for newer approaches: as we will see in later chapters, many fancy statistical
learning approaches can be seen as generalizations or extensions of linear regression, such as
Generalized Linear Models.

## Definition

Suppose that a random sample from size $n$ is drawn from a population, and measurements $(x_{i1}, x_{i2}...x_{ik}, y_i)$ where $i = 1, ... , n$, are obtained on the n individuals.

The random variables $x_{i1}, x_{i2}...x_{ik}$ are commonly termed as *predictor variables* (or simply, predictors), but depending on the field of application, they may also be called *independent variables*, *regressors*, *features*, *covariates* or *explanatory variables*.

The $y_i$ variable is called the *response variable* (or simply, response). Other terms include *target variable*, *variate*, *dependent variable* or *outcome variable*.

The Linear Regression Model represents a relation between the response variable $y$ and the predictor variables $x_{1}, x_{2}...x_{k}$ (with the lowercase notation for simplicity) of the form:

$$ 
\begin{align}
y = \beta_0 + \beta_1 + ... + \beta_k + \epsilon
\end{align} 
$$

Where $\beta_0, ..., \beta_k$ are constants *regression coefficients*, and $\epsilon$ is a random error term that follows a normal distribution with mean zero ($\mu = 0$) and constant variance $\sigma^2$. That is, $\epsilon \sim Norm(\mu = 0, \sigma^2)$. Also, the random errors are assumed to be independent for different individuals of the sample.

The parameters of the model $\beta_0, ..., \beta_k$, and the variance $\sigma^2$ are unknown and have to be estimated from the sample data.

Note that the relationship between the predictors and the response is not necessarily linear, as polynomial or interaction terms may be included, but it is necessarily linear in the beta coefficients. That is, the relationship is modeled as a linear combination of the parameters.

Note that, in the general linear regression model, the response variable y has a normal distribution with the mean:

$$ 
\begin{align}
\mathbb{E}(y) = \beta_0 + \beta_1 \cdot x_1 + ... + \beta_k \cdot x_k
\end{align} 
$$

# Normal Equation

The analytical solution to the parameter vector of the Linear Regression Model
is called the **Normal Equation**.
It is given by:
$\hat{\beta} = (X^T X)^{-1} \cdot  X^T \cdot y$

Though, it is actually unusually to calculate the parameters of the
Linear Regression model using the Normal Equation, as certain, faster algorithms are preferred
(such as SVD / LU Decomposition / Gradient Descent).

## Categorical Variables

In a Regression Model, categorical variables are encoded. 
There are two main encoding methods: **one-hot encoding** and **dummy encoding**.

In one hot-encoding a category like `['BMW', 'AUDI', 'Volvo']` would be represented using three variables:
- `BMW = [1, 0, 0]`
- `AUDI=[0, 1, 0]` 
- `Volvo = [0, 0, 1]`. 

In dummy-encoding, we'd use the following strategy:
- `AUDI=[1, 0]` 
- `Volvo = [0, 1]`.

Notice there's no representation of BMW, as `BMW` is called the *"reference category"*
and is represented by `BMW=[0, 0]`.


