
rm(list = ls(all = TRUE))

setwd("C:\\Users\\Nikhil Chalakkal\\Documents\\Datasets\\Machine Learning Final Project\\crossvalidation")

LogLossBinary = function(actual, predicted, eps = 1e-15) {
  predicted = pmin(pmax(predicted, eps), 1-eps)
  - (sum(actual * log(predicted) + (1 - actual) * log(1 - predicted))) / length(actual)
}

val_data <- read.csv("C:\\Users\\Nikhil Chalakkal\\Documents\\Datasets\\Machine Learning Final Project\\train subsets\\part319.csv")
head(val_data)

ID <- seq(1, nrow(val_data), 1)

y_pred_lasso <- read.csv("y_pred_lasso_regression.csv")
names(y_pred_lasso) <- c("ID", "Prob_Lasso")

y_pred_ridge <- read.csv("y_pred_ridge_regression.csv")
names(y_pred_ridge) <- c("ID", "Prob_Ridge")

y_pred_lasso_cv <- read.csv("y_pred_lasso_regression_cv.csv")
names(y_pred_lasso_cv) <- c("ID", "Prob_Lasso_CV")

y_pred_ridge_cv <- read.csv("y_pred_ridge_regression_cv.csv")
names(y_pred_ridge_cv) <- c("ID", "Prob_Ridge_CV")

y_pred_random_forest <- read.csv("y_pred_random_forest.csv")
names(y_pred_random_forest) <- c("ID", "Prob_Random_Forest")
y_pred_random_forest$ID <- ID

y_pred_logistic <- read.csv("y_pred_logistic_regression.csv")
y_pred_logistic$ID <- ID
y_pred_logistic <- y_pred_logistic[,c(2,1)]
names(y_pred_logistic) <- c("ID", "Prob_Logistic")

ensemble_values <- as.data.frame(matrix(data = NA, nrow = nrow(val_data), ncol = 1))
ensemble_values$V1 <- ID
names(ensemble_values)[1] <- "ID"
ensemble_values$Y_Actual <- val_data$click
ensemble_values <- merge(ensemble_values, y_pred_lasso, by.x = "ID", by.y = "ID", all.x = TRUE)
ensemble_values <- merge(ensemble_values, y_pred_ridge, by.x = "ID", by.y = "ID", all.x = TRUE)
ensemble_values <- merge(ensemble_values, y_pred_logistic, by.x = "ID", by.y = "ID", all.x = TRUE)
ensemble_values <- merge(ensemble_values, y_pred_random_forest, by.x = "ID", by.y = "ID", all.x = TRUE)
ensemble_values <- merge(ensemble_values, y_pred_lasso_cv, by.x = "ID", by.y = "ID", all.x = TRUE)
ensemble_values <- merge(ensemble_values, y_pred_ridge_cv, by.x = "ID", by.y = "ID", all.x = TRUE)

ensemble_values$Prob_Ensemble <- 0
ensemble_values$Prob_Ensemble <- apply(ensemble_values[,c(3:5)], 1, FUN = mean)

colnames(ensemble_values)
head(ensemble_values)

LogLossBinary(ensemble_values$Y_Actual, ensemble_values$Prob_Ensemble)
