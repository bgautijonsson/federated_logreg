
# Federated Learning for Logistic Regression with pytorch

<!-- badges: start -->
<!-- badges: end -->

This is a simplified example of federated learning if we ignore the clients running their own server and the differential privacy.

prep_data.R creates the 30 train and test data sets

GPT4_torch.py defines and runs code for a simplified version of federated learning to fit a logistic regression model in each of the 30 datasets. each iteration samples 5 datasets, calculated gradients and updates all of the 30 models using the average of the gradients

