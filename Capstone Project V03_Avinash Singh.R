

library(caTools) # Split Data into Test and Train Set
library(caret) # for confusion matrix function
library(randomForest) # to build a random forest model
library(rpart) # to build a decision model
library(rpart.plot) # to plot decision tree model
library(rattle) 
library(xgboost) # to build a XGboost model
library(DMwR) # for sMOTE
library(tidyverse)
library(caret)
library(reshape2)
library(RColorBrewer)
library(summarytools) 
library(DataExplorer)
library(janitor)
library(visdat)
library(naniar)
library(caTools) # Split Data into Test and Train Set
library(caret) # for confusion matrix function
library(randomForest) # to build a random forest model
library(rpart) # to build a decision model
library(rpart.plot) # to plot decision tree model
library(rattle) 
library(xgboost) # to build a XGboost model
library(DMwR) # for sMOTE
library(solitude)

library(janitor)
library(dplyr)
library(cluster)
library(caret)


setwd("C:/Users/a201sing/OneDrive - Nokia/HP Laptop backup  D drive/Drive D/PGP DSBA/Capstone project")
CP3 = read.csv("CP3_Cleaned")

str(CP3)

CP3[sapply(CP3, is.integer)] = lapply(CP3[sapply(CP3, is.integer)], as.numeric)
CP3[sapply(CP3, is.character)] = lapply(CP3[sapply(CP3, is.character)], as.factor)

str(CP3)
CP3 = CP3[,-1]

CP3_FS = CP3

write.csv(CP3_FS, file = "CP3_FS_Cleaned data for submission 3")

CP3_FS1 = CP3_FS

## Split the transformed data in Train and test data

set.seed(1234)

CP3_split = sample.split(CP3_FS1$satisfaction, SplitRatio=0.70)
train_CP3_FS1 = subset(CP3_FS1, CP3_split ==T)
test_CP3_FS1 = subset(CP3_FS1, CP3_split ==F)

## Check split consistency
prop.table(table(train_CP3_FS1$satisfaction))
prop.table(table(test_CP3_FS1$satisfaction))
prop.table(table(CP3_FS1$satisfaction))

## for future reusability
write.csv(train_CP3_FS1, file = "CP3_Train Data")
write.csv(test_CP3_FS1, file = "CP3_Test Data")

train_CP3_FS2 = train_CP3_FS1
test_CP3_FS2 = test_CP3_FS1

###----------------####

# Setting up the general parameters for training multiple models
?trainControl
# Define the training control
fitControl1 <- trainControl(
  method = 'repeatedcv',           # k-fold cross validation
  number = 3,                      # number of folds or k
  repeats = 1,                     # repeated k-fold cross-validation
  allowParallel = TRUE,
  classProbs = TRUE
) 

attach(train_CP3_FS2)

set.seed(1234)

# Model _1 : GLM : Simple Logistic Regression Model
lr_model <- train(satisfaction ~ ., data = train_CP3_FS2,
                  method = "glm",
                  family = "binomial",
                  trControl = fitControl1)

summary(lr_model)

library(devtools) 
install_github("uc-cfss/rcfss")
require(rcfss)


logit2prob(coef(lr_model))

set.seed(1234)

lr1_model = glm(satisfaction ~ ., data = train_CP3_FS2,family = "binomial")

summary(lr1_model)
exp(coef(lr1_model))
lrres = predict(lr1_model, train_CP3_FS2, type = "response")

library(ROCR)
ROCRpredlr = prediction(lrres, satisfaction)
ROCRperflr = performance(ROCRpredlr, "tpr", "fpr")

plot(ROCRperflr, colorize = TRUE, print.cutoffs.at = seq(0.1, by = 0.1))

# Predict using the trained model & check performance on test set
lr_predictions_test <- predict(lr_model, newdata = test_CP3_FS2, type = "raw")
confusionMatrix(lr_predictions_test, test_CP3_FS2$satisfaction)
summary(lr_predictions_test)


lr1_predictions_test <- predict(lr1_model, newdata = test_CP3_FS2, type = "response")
confusionMatrix(lr1_predictions_test, test_CP3_FS2$satisfaction, type)

set.seed(1234)

# Model_2 : Naive-Bayes
library(naivebayes)
nb_model <- train(satisfaction ~ ., data = train_CP3_FS2,
                  method = "naive_bayes",
                  trControl = fitControl1)

summary(nb_model)


# Predict using the trained model & check performance on test set

nb_predictions_test <- predict(nb_model, newdata = test_CP3_FS2, type = "raw")
confusionMatrix(nb_predictions_test, test_CP3_FS2$satisfaction)

set.seed(1234)

# Model_3 : KNN 
knn_model <- train(satisfaction ~ ., data = train_CP3_FS2,
                   preProcess = c("center", "scale"),
                   method = "knn",
                   tuneLength = 3,
                   trControl = fitControl1) 


# Predict using the trained model & check performance on test set

knn_predictions_test <- predict(knn_model, newdata = test_CP3_FS2, type = "raw")
confusionMatrix(knn_predictions_test, test_CP3_FS2$satisfaction)

# Model_4 :  Rpart : Single CART decision tree 

# Define the training control

?trainControl

fitControl <- trainControl(
  method = 'repeatedcv',           # k-fold cross validation
  number = 3,                      # number of folds or k
  repeats = 1,                     # repeated k-fold cross-validation
  allowParallel = TRUE,
  classProbs = TRUE,
  summaryFunction=twoClassSummary  # should class probabilities be returned
) 

set.seed(1234)

dtree_model <- train(satisfaction ~ ., data = train_CP3_FS2,
                     method = "rpart",
                     minbucket = 1000,
                     cp = 0.002,
                     tuneLength = 10,
                     trControl = fitControl1)

dtree_model
# Plot the cp vs ROC values to see the effect of cp on ROC 

# Plot the CP values 
plot(dtree_model)

# Plot the tree
?fancyRpartPlot
fancyRpartPlot(dtree_model$finalModel,cex=0.5, uniform = TRUE)


# Predict using the trained model & check performance on test set
?predict
dtree_predictions_test <- predict(dtree_model , newdata = test_CP3_FS2, type = "raw")
confusionMatrix(dtree_predictions_test, test_CP3_FS2$satisfaction)

## Build a CART model on the train dataset with new method as the old method shows error as : Error in eval(predvars, data, env) : object 'genderMale' not found

##We will use the "rpart" and the "rattle" libraries to build decision trees.

r.ctrl = rpart.control(minsplit = 50, minbucket = 10, cp = 0, xval = 10)

decision_tree1 = rpart(formula = satisfaction~., data = train_CP3_FS2, method = "class", control = r.ctrl)
decision_tree1

printcp(decision_tree1)
plotcp(decision_tree1)
rpart.plot(decision_tree1)

dtree_class_predictions_test <- predict(decision_tree1, newdata = test_CP3_FS2, type = "class")

confusionMatrix(predict(decision_tree1, newdata = test_CP3_FS2, type = "class"),test_CP3_FS2$satisfaction)

set.seed(1234)
pdecision_tree1 = prune(decision_tree1, cp= 0.0019,"CP")
printcp(pdecision_tree1)
plotcp(pdecision_tree1)
pdecision_tree1


rpart.plot(pdecision_tree1, cex = 0.6, uniform = TRUE, type = 2)

fancyRpartPlot(pdecision_tree1,cex=0.6, uniform = TRUE)

path.rpart(pdecision_tree1, c(16,24,70))

pdecision_tree1$variable.importance


pdtree_class_predictions_test <- predict(pdecision_tree1, newdata = test_CP3_FS2, type = "class")

confusionMatrix(predict(pdecision_tree1, newdata = test_CP3_FS2, type = "class"),test_CP3_FS2$satisfaction)


# Model_5 : Random Forest 
rf_model <- train(satisfaction ~ ., data = train_CP3_FS2,
                  method = "rf",
                  ntree = 30,
                  maxdepth = 5,
                  tuneLength = 10,
                  trControl = fitControl1)
print(rf_model)

set.seed(1234)
CP3_RFmodel = randomForest(satisfaction~.,train_CP3_FS2,ntree=150,mtry=8,nodesize=2,importance=TRUE)
print(CP3_RFmodel)
plot(CP3_RFmodel)
varImpPlot(CP3_RFmodel, type = 1)

CP3_RFmodel_1 = randomForest(satisfaction~.,train_CP3_FS2,ntree=50,mtry=18,nodesize=10,importance=TRUE)
print(CP3_RFmodel_1)
plot(CP3_RFmodel_1)
varImpPlot(CP3_RFmodel_1, type = 1)
table(varImpPlot(CP3_RFmodel_1, type = 1))
list(varImpPlot(CP3_RFmodel_1, type = 1))


# Predict using the trained model & check performance on test set
rf_predictions_test <- predict(rf_model, newdata = test_CP3_FS2, type = "raw")
confusionMatrix(rf_predictions_test, test_CP3_FS2$satisfaction)

# Predict using the trained model & check performance on test set
rf_predictions_test_1 <- predict(CP3_RFmodel, newdata = test_CP3_FS2, type = "response")
confusionMatrix(rf_predictions_test_1, test_CP3_FS2$satisfaction)

##For RFmodel_1
confusionMatrix(predict(CP3_RFmodel_1, newdata = test_CP3_FS2, type = "class"),test_CP3_FS2$satisfaction)

# Model_6 : Gradient Boosting Machines 
gbm_model <- train(satisfaction ~ ., data = train_CP3_FS2,
                   method = "gbm",
                   trControl = fitControl1,
                   verbose = FALSE)


# Predict using the trained model & check performance on test set
gbm_predictions_test <- predict(gbm_model, newdata = test_CP3_FS2)
confusionMatrix(gbm_predictions_test, test_CP3_FS2$satisfaction)


# Model_7 : Xtreme Gradient boosting Machines [without smote or with highly unbalanced data]
cv.ctrl <- trainControl(method = "repeatedcv", repeats = 1,number = 3, 
                        
                        classProbs = TRUE,
                        allowParallel=T)

xgb.grid <- expand.grid(nrounds = 100,
                        eta = c(0.01),
                        max_depth = c(2,4),
                        gamma = 0,               #default=0
                        colsample_bytree = 1,    #default=1
                        min_child_weight = 1,    #default=1
                        subsample = 1            #default=1
)

xgb_model <-train(satisfaction~.,
                  data=train_CP3_FS2,
                  method="xgbTree",
                  trControl=cv.ctrl,
                  tuneGrid=xgb.grid,
                  verbose=T,
                  nthread = 2
)


# Predict using the trained model & check performance on test set
xgb_predictions_test <- predict(xgb_model, newdata = test_CP3_FS2, type = "raw")
confusionMatrix(xgb_predictions_test, test_CP3_FS2$satisfaction)


#---------------------------  COMPARING MODELS  ---------------------
# Compare model performances using resample()
models_to_compare <- resamples(list(Logistic_Regression = lr_model, 
                                    Naive_Bayes = nb_model, 
                                    KNN = knn_model, 
                                    CART_Decision_tree = dtree_model, 
                                    Random_Forest = rf_model,
                                    Gradient_boosting = gbm_model,
                                    eXtreme_gradient_boosting = xgb_model
))


# Summary of the models performances
summary(models_to_compare)


# Draw box plots to compare models

scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(models_to_compare, scales=scales)

set.seed(1234)

# Bagging
library(gbm)
library(xgboost)
library(caret)
library(ipred)
library(plyr)
library(rpart)
library(dplyr)
mod.bagging <- bagging(satisfaction ~.,
                       data=train_CP3_FS2,
                       control=rpart.control(maxdepth=10, minsplit=10))

bag.pred <- predict(mod.bagging, test_CP3_FS2)

confusionMatrix(test_CP3_FS2$satisfaction,bag.pred)


# Boosting
#mod.boost <- gbm(formula = satisfaction ~ .,distribution = "bernoulli",
#                data = train_CP3_FS2,  n.trees = 100,
#                 interaction.depth = 1,  shrinkage = 0.01,
#                 cv.folds = 1,  n.cores = 1,  verbose = FALSE)


mod.boost <- gbm((unclass(satisfaction)-1) ~ .,data=train_CP3_FS2, distribution=
                   "bernoulli",n.trees =5000 , interaction.depth =10, shrinkage=0.05, cv.folds = 1,  n.cores = 1,  verbose = FALSE)

summary(mod.boost)
#boost.pred <- predict(mod.boost, test_CP3_FS2)
boost.pred <- predict(mod.boost, test_CP3_FS2,n.trees =5000, type="response")

y_pred_num <- ifelse(boost.pred > 0.5, 1, 0)
y_pred <- factor(y_pred_num, levels=c(0, 1))
confusionMatrix(test_CP3_FS2$satisfaction,boost.pred)


