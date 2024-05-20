# Price-Optimization-Ecommerce
price optimization, price and demand are considered under turnover, assumption curve

## Hypothesis 1 - Regression based on QP(turnover) as one variable
reference: https://medium.com/academy-team/price-optimization-with-machine-learning-the-impact-of-data-science-on-commercial-strategies-a87b6dbe95e0
math hypothesis in the backend
![image](https://github.com/ShawnLiu119/Price-Optimization-Ecommerce/assets/43327902/00aaaf6c-7675-4e73-9247-75c5e5d02b6b)

**pros: consider Q into the variable so price elasticity is factored in**
**cons: Q and P is not simply inverse to each other. P = Optimal Turnover / Q is a little bit simple

## Hypothesis 2 - Purchase Probability Prediction (ML classcification) + Profit Optimization (Operation Research)

Unfortunately Gekko does not currently support xgboost or other tree-based methods. Because the back-end solvers use gradient descent based methods, the decision functions of models need to be diffentiable and rewritable with Gekko variables. So far, Gaussian process regression, support vector regression, linear regression, and neural networks have been integrated into Gekko: https://gekko.readthedocs.io/en/latest/ml.html. 
