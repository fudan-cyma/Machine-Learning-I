
library(ISLR)
library(glmnet)

rm(list = ls(all = TRUE))

setwd("C:\\Users\\Nikhil Chalakkal\\Documents\\Datasets\\Machine Learning Final Project\\milliontrain")

LogLossBinary = function(actual, predicted, eps = 1e-15) {
  predicted = pmin(pmax(predicted, eps), 1-eps)
  - (sum(actual * log(predicted) + (1 - actual) * log(1 - predicted))) / length(actual)
}

train_data <- rbind(read.csv("part1.csv"), read.csv("part2.csv"), read.csv("part3.csv"), read.csv("part4.csv"), read.csv("part5.csv"))
val_data <- read.csv("C:\\Users\\Nikhil Chalakkal\\Documents\\Datasets\\Machine Learning Final Project\\train subsets\\part319.csv")

train_data = as.data.frame(lapply(train_data, factor))
train_data$click = as.numeric(as.character(train_data$click))

val_data = as.data.frame(lapply(val_data, factor))
val_data$click = as.numeric(as.character(val_data$click))

x_train <- model.matrix(train_data$click ~ ., train_data[,c(2:22)])
x_val <- model.matrix(val_data$click ~ ., val_data[,c(2:22)])

y_train <- train_data$click
y_val_act <- val_data$click

grid <- 10^seq(7, -7, length = 500)

# Ridge_Manual_Dummy

model_ridge <- glmnet(x_train, y_train, alpha = 0, lambda = grid)
y_val_pred_matrix_ridge <- predict(model_ridge, newx = x_val)

loss_matrix_ridge = as.vector(matrix(data = 0, nrow = ncol(y_val_pred_matrix_ridge), ncol = 1))
for (i in 1:ncol(y_val_pred_matrix_ridge)){
  loss_matrix_ridge[i] <- LogLossBinary(y_val_act, y_val_pred_matrix_ridge[,c(i)])
}

logloss_val_ridge <- min(loss_matrix_ridge)
best_lambda_ridge <- model_ridge$lambda[which.min(loss_matrix_ridge)]
best_ypred_ridge <- y_val_pred_matrix_ridge[,c(which.min(loss_matrix_ridge))]

remove(y_val_pred_matrix_ridge)

cat("With Dummy Variables\nLog Loss Ridge ", logloss_val_ridge)

# Lasso_Manual_Dummy

model_lasso <- glmnet(x_train, y_train, alpha = 1, lambda = grid)
y_val_pred_matrix_lasso <- predict(model_lasso, newx = x_val)

loss_matrix_lasso = as.vector(matrix(data = 0, nrow = ncol(y_val_pred_matrix_lasso), ncol = 1))
for (i in 1:ncol(y_val_pred_matrix_lasso)){
  loss_matrix_lasso[i] <- LogLossBinary(y_val_act, y_val_pred_matrix_lasso[,c(i)])
}

logloss_val_lasso <- min(loss_matrix_lasso)
best_lambda_lasso <- model_lasso$lambda[which.min(loss_matrix_lasso)]
best_ypred_lasso <- y_val_pred_matrix_lasso[,c(which.min(loss_matrix_lasso))]

remove(y_val_pred_matrix_lasso)

cat("With Dummy Variables\nLog Loss Lasso ", logloss_val_lasso)

remove(x_train)

write.csv(best_ypred_ridge,"C:\\Users\\Nikhil Chalakkal\\Dropbox\\Emory MSBA\\Courses\\ISOM 674\\9.0 Final Project\\y_pred_ridge.csv")
write.csv(best_ypred_lasso,"C:\\Users\\Nikhil Chalakkal\\Dropbox\\Emory MSBA\\Courses\\ISOM 674\\9.0 Final Project\\y_pred_lasso.csv")
