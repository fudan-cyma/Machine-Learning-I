
rm(list = ls(all = TRUE))

setwd("C:\\Users\\Nikhil Chalakkal\\Documents\\Datasets\\Machine Learning Final Project\\train subsets")

train_data <- read.csv("part1.csv")
val_data <- read.csv("part2.csv")

colnames(train_data)
dim(train_data)

train_data = as.data.frame(lapply(train_data, factor))
train_data$click = as.numeric(as.character(train_data$click))

colnames(val_data)
dim(val_data)

val_data = as.data.frame(lapply(val_data, factor))
val_data$click = as.numeric(as.character(val_data$click))

prob_fn <- function(x) {
  probs <- table(x)
  probs <- probs / sum(probs)
  return(probs)
}

PClick <- mean(train_data$click)
PGivenClick <- lapply(train_data[train_data$click == 1,], prob_fn)
PGivenNoClick <- lapply(train_data[train_data$click != 1,], prob_fn)

pred_fn <- function(DataRow, PGivenClick, PGivenNoCilck, PClick) {
  t1 <- 1
  t2 <- 1
  for(x in names(DataRow)) {
    t1 <- t1 * PGivenClick[[x]][DataRow[x]]
    t2 <- t2 * PGivenNoClick[[x]][DataRow[x]]
  }
  out <- (t1 * PClick) / ((t1 * PClick) + (t2 * (1 - PClick)))
  return(out)
}

PClick_val_Pred <- apply(val_data[,c(2:22)], 1, pred_fn, PGivenClick, PGivenNoClick, PClick)

PClick_Val_Act <- val_data[,c(1)]

system.time(PClick_val <- apply(val_data[,c(2:22)], 1, pred_fn, PGivenClick, PGivenNoClick, PClick))

LogLossBinary = function(actual, predicted, eps = 1e-15) {
  predicted = pmin(pmax(predicted, eps), 1-eps)
  - (sum(actual * log(predicted) + (1 - actual) * log(1 - predicted))) / length(actual)
}

LogLoss_Val_NB <- LogLossBinary(PClick_Val_Act, PClick_val_Pred)

