
# Voter Engagement Analytics

# removing all objects (clearing platform)
rm(list = ls())
set.seed(123) # to get same results 

library(tidyverse) # data wrangling, plotting
library(caret)  # data splitting, preparation
library(rpart)  # decision trees 
library(forcats) # categorical to factorial


# 1) Loading data

voter <- read.csv(
  "C:/Users/vemul/Desktop/CSDA 6010 CASE 1/VoterPersuasion.csv",
  stringsAsFactors = FALSE
)

# Inspecting structure and size
str(voter)
dim(voter)

# Target variable
voter$MOVED_A <- factor(voter$MOVED_A, levels = c(0, 1))

# 2) DATA CLEANING & PREPROCESSING
# Droping or removing variables (leakage variables)
drop_cols <- intersect(
  names(voter),
  c("VOTER_ID", "Unnamed.0", "opposite", "MOVED_AD")
)
if (length(drop_cols) > 0) {
  voter <- voter %>% select(-all_of(drop_cols))
}

# Convert character -> factor
voter <- voter %>% mutate(across(where(is.character), as.factor))

# 3) BASIC EXPLORATION

# Outcome balance
table(voter$MOVED_A)
prop.table(table(voter$MOVED_A))

# Party distribution
colSums(voter[, c("PARTY_D", "PARTY_R", "PARTY_I")])

# Gender distribution
colSums(voter[, c("GENDER_F", "GENDER_M")])

# VISUAL EXPLORATION

# a. Target variable distribution
barplot(
  table(voter$MOVED_A),
  main = "Distribution of Voter Persuasion Outcome (MOVED_A)",
  xlab = "Persuasion Outcome",
  ylab = "Number of Voters"
)

# b. Age vs persuasion outcome
boxplot(
  AGE ~ MOVED_A,
  data = voter,
  main = "Age Distribution by Persuasion Outcome",
  xlab = "Persuasion Outcome (MOVED_A)",
  ylab = "Age"
)

# c. Party affiliation vs persuasion
party_group <- ifelse(
  voter$PARTY_D == 1, "Democrat",
  ifelse(voter$PARTY_R == 1, "Republican", "Independent")
)

party_table <- table(voter$MOVED_A, party_group)

barplot(
  party_table,
  beside = TRUE,
  legend = TRUE,
  main = "Persuasion Outcome by Party Affiliation",
  xlab = "Party Affiliation",
  ylab = "Number of Voters"
)

# d. Voting participation vs persuasion
boxplot(
  G_PELIG ~ MOVED_A,
  data = voter,
  main = "General Election Participation Likelihood by Persuasion Outcome",
  xlab = "Persuasion Outcome (MOVED_A)",
  ylab = "General Election Participation Probability"
)

# 4) Training and Testing data split 80:20
train_index <- createDataPartition(voter$MOVED_A, p = 0.80, list = FALSE)
train_raw <- voter[train_index, ]
test_raw  <- voter[-train_index, ]

train_raw$MOVED_A <- factor(train_raw$MOVED_A, levels = c(0, 1))
test_raw$MOVED_A  <- factor(test_raw$MOVED_A,  levels = c(0, 1))

# 5) Missing handling (factors: NA -> "Missing")
train_raw <- train_raw %>%
  mutate(across(where(is.factor), ~ fct_explicit_na(.x, na_level = "Missing")))
test_raw <- test_raw %>%
  mutate(across(where(is.factor), ~ fct_explicit_na(.x, na_level = "Missing")))

# 6) Dummy encoding predictors only
# dummyVars creates one-hot encoded numeric features for models
dv <- dummyVars(~ ., data = train_raw %>% select(-MOVED_A), fullRank = TRUE)
x_train <- as.data.frame(predict(dv, newdata = train_raw %>% select(-MOVED_A)))
x_test  <- as.data.frame(predict(dv, newdata = test_raw  %>% select(-MOVED_A)))

# Extracting target labels
y_train <- train_raw$MOVED_A
y_test  <- test_raw$MOVED_A

# caret labels
y_train2 <- factor(ifelse(y_train == "1", "Yes", "No"), levels = c("No", "Yes"))
y_test2  <- factor(ifelse(y_test  == "1", "Yes", "No"), levels = c("No", "Yes"))

# 7) Imputing numeric missing 
pp <- preProcess(x_train, method = "medianImpute")
x_train <- predict(pp, x_train)
x_test  <- predict(pp, x_test)

# 8) Removing near-zero variance
nzv <- nearZeroVar(x_train)
if (length(nzv) > 0) {
  x_train <- x_train[, -nzv, drop = FALSE]
  x_test  <- x_test[, -nzv, drop = FALSE]
}

# 9) Cross Validation control
ctrl <- trainControl(
  method = "repeatedcv",    # repeated k-fold cross validation
  number = 5, # here data split into 5 parts
  repeats = 3, # repeats 5 times 
  classProbs = TRUE, # class probabilities
  summaryFunction = twoClassSummary # model performance is measured
)

# 10) Logistic Regression

set.seed(123)
logit_fit <- train(
  x = x_train, y = y_train2,  # Predictors & outcome
  method = "glm",       # generalized linear model
  family = binomial(),  # logistic regression (binary outcome)
  metric = "ROC",  # for Performance evaluation
  trControl = ctrl
)

logit_prob <- predict(logit_fit, x_test, type = "prob")[, "Yes"]
logit_pred <- factor(ifelse(logit_prob >= 0.5, "Yes", "No"), levels = c("No", "Yes"))
cm_logit <- confusionMatrix(logit_pred, y_test2)
cm_logit

# Predicted probability outcomes
hist(
  logit_prob,
  breaks = 20,
  main = "Predicted Probability of Persuasion (Logistic Regression)",
  xlab = "Predicted Probability",
  ylab = "Number of Voters"
)

# 11) Decision Tree (tuned + constrained)
set.seed(123)
tree_grid <- expand.grid(cp = seq(0.001, 0.05, by = 0.002))
tree_fit <- train(
  x = x_train, y = y_train2,  # predictor & outcome
  method = "rpart",           # tree using rpart package
  tuneGrid = tree_grid,
  metric = "ROC",           # performance metric
  trControl = ctrl, 
  control = rpart.control(maxdepth = 4, minsplit = 50)
)

tree_prob <- predict(tree_fit, x_test, type = "prob")[, "Yes"]
tree_pred <- factor(ifelse(tree_prob >= 0.5, "Yes", "No"), levels = c("No", "Yes"))
cm_tree <- confusionMatrix(tree_pred, y_test2)
cm_tree

## Plotting the tree
install.packages("rpart.plot")
library(rpart.plot)
rpart.plot(tree_fit$finalModel,
           type = 2,
           extra = 104,
           fallen.leaves = TRUE,
           main = "Decision Tree for Voter Persuasion")


# 12) Print results
print("Confusion Matrix of Logistic Regression")
print(cm_logit)

print("Confusion Matrix of Decision Tree")
print(cm_tree)

results <- data.frame(
  Model = c("Logistic Regression", "Decision Tree"),
  Accuracy = c(cm_logit$overall["Accuracy"], cm_tree$overall["Accuracy"]),
  Sensitivity = c(cm_logit$byClass["Sensitivity"], cm_tree$byClass["Sensitivity"]),
  Specificity = c(cm_logit$byClass["Specificity"], cm_tree$byClass["Specificity"])
)

print("Model Performance Comparison")
print(results)

barplot(
  results$Accuracy,
  names.arg = results$Model,
  ylim = c(0, 1.5),
  main = "Model Accuracy Comparison (80:20 Split)",
  ylab = "Accuracy"
)

#  Random Forest model

library(randomForest)
library(e1071)
# install.packages("kernlab")
library(kernlab)
# install.packages("pROC")
library(pROC)

# install.packages("ranger")
library(ranger)

# Faster CV
ctrl_fast <- trainControl(
  method = "cv",
  number = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

set.seed(123)
rf_fit <- train(
  x = x_train, y = y_train2,
  method = "ranger",
  metric = "ROC",
  trControl = ctrl_fast,
  tuneGrid = data.frame(
    mtry = floor(sqrt(ncol(x_train))),
    splitrule = "gini",
    min.node.size = 10
  ),
  num.trees = 200
)

rf_prob <- predict(rf_fit, x_test, type = "prob")[, "Yes"]
rf_pred <- factor(ifelse(rf_prob >= 0.5, "Yes", "No"), levels = c("No","Yes"))
cm_rf <- confusionMatrix(rf_pred, y_test2)
cm_rf

##
results <- data.frame(
  Model = c("LR", "DT", "RF"),
  Accuracy = c(cm_logit$overall["Accuracy"], cm_tree$overall["Accuracy"],
               cm_rf$overall["Accuracy"]),
  Sensitivity = c(cm_logit$byClass["Sensitivity"], cm_tree$byClass["Sensitivity"],
                  cm_rf$byClass["Sensitivity"]),
  Specificity = c(cm_logit$byClass["Specificity"], cm_tree$byClass["Specificity"],
                  cm_rf$byClass["Specificity"])
)

print(results)

barplot(
  results$Accuracy,
  names.arg = results$Model,
  ylim = c(0, 1),
  main = "Model Accuracy Comparison (80:20 Split)",
  ylab = "Accuracy",
  las = 2
)





