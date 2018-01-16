
library(ISLR)
library(glmnet)

rm(list = ls(all = TRUE))

setwd("C:\\Users\\Nikhil Chalakkal\\Documents\\Datasets\\Machine Learning Final Project\\milliontrain")

LogLossBinary = function(actual, predicted, eps = 1e-15) {
  predicted = pmin(pmax(predicted, eps), 1-eps)
  - (sum(actual * log(predicted) + (1 - actual) * log(1 - predicted))) / length(actual)
}

train_data <- rbind(read.csv("part1.csv"), read.csv("part2.csv"), read.csv("part3.csv"))
val_data <- read.csv("C:\\Users\\Nikhil Chalakkal\\Documents\\Datasets\\Machine Learning Final Project\\train subsets\\part319.csv")

train_data = as.data.frame(lapply(train_data, factor))
train_data$click = as.numeric(as.character(train_data$click))

val_data = as.data.frame(lapply(val_data, factor))
val_data$click = as.numeric(as.character(val_data$click))

x_train <- model.matrix(train_data$click ~ ., train_data[,c(2:22)])
x_val <- model.matrix(val_data$click ~ ., val_data[,c(2:22)])

y_train <- train_data$click
y_val_act <- val_data$click

# Ridge_CV_Dummy

model_ridge_cv <- cv.glmnet(x_train, y_train, alpha = 0)
y_val_pred_ridge_cv <- predict(model_ridge_cv, newx = x_val)
logloss_val_ridge_CV <- LogLossBinary(y_val_act, y_val_pred_ridge_cv)

cat("With Dummy Variables\nLog Loss Ridge ", logloss_val_ridge_CV)

# Lasso_CV_Dummy

model_lasso_cv <- cv.glmnet(x_train, y_train, alpha = 1)
y_val_pred_lasso_cv <- predict(model_lasso_cv, newx = x_val)
logloss_val_lasso_CV <- LogLossBinary(y_val_act, y_val_pred_lasso_cv)

cat("With Dummy Variables\nLog Loss Lasso ", logloss_val_lasso_CV)

remove(x_train)

write.csv(y_val_pred_ridge_cv,"C:\\Users\\Nikhil Chalakkal\\Dropbox\\Emory MSBA\\Courses\\ISOM 674\\9.0 Final Project\\y_pred_ridge_cv.csv")
write.csv(y_val_pred_lasso_cv,"C:\\Users\\Nikhil Chalakkal\\Dropbox\\Emory MSBA\\Courses\\ISOM 674\\9.0 Final Project\\y_pred_lasso_cv.csv")
