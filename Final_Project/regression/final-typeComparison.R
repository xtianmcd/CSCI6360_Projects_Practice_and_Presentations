# classification on wine dataset. Type 0 is white; type 1 is red
library(foreign)

library(caret)
library(MASS)
library(bnlearn)
library(bnclassify)
library(class)
library(naivebayes)
library(e1071)
library(dplyr)
library(graph)

dataset <- read.csv("/Users/yuanmingshi/downloads/winetypecomparison.csv")
# dataset <- dataset[, !(colnames(dataset) %in% c('type'))]

# using the boilerplte code provided by Bahaa
score <- function(expected, predicted){
  tp = length(expected[which(expected == '1' & predicted == '1')])   # adding 1 to avoid NaN
  fp = length(expected[which(expected == '0' & predicted == '0')])    # adding 1 to avoid NaN
  fn = length(expected[which(expected == '1' & predicted == '1')])  # adding 1 to avoid NaN
  tn = length(expected[which(expected == '0' & predicted == '0')])    # adding 1 to avoid NaN
  
  accuracy = (tp + tn) / (length(expected))
  recall = tp / (tp + fn + 1)
  precision = tp / (tp + fp + 1)
  f.score = (2 * tp) / (2 * tp + fn + fp)
  
  r <- cbind(accuracy, recall, precision, f.score) 
  colnames(r)<- c('accuracy','recall','precision','fscore') 
  return(r)
}

trainAndTestFold <- function(fold, dataset) { 
  
  # Model 1: Naive Bayes
  dataset[] <- lapply(dataset, as.factor)
  training = dataset[-fold,]
  testing = dataset[fold,]
  model1 = naiveBayes(training$type ~., data=training)
  model1.pred = predict(model1, testing)
  model1.scores = score(testing$type, model1.pred)
  
  # Model 2: TAN Bayes
  model2 = bnc('tan_cl', 'type', training, smooth = 1)
  model2.pred = predict(model2, testing)
  model2.scores = score(testing$type, model2.pred)
  
  # Model 3: LDA (Thanks Sean for reminding me the dataframe setting)
  dataset <- read.csv("/Users/yuanmingshi/downloads/winetypecomparison.csv")
  dataset <- as.data.frame(dataset)
  training = dataset[-fold,]
  testing = dataset[fold,]
  model3 = lda(formula = training$type ~., data=training)
  model3.pred = predict(model3, testing)
  model3.scores = score(testing$type, model3.pred$class)
  
  # Model 4: Logistic Regression
  # From https://www.r-bloggers.com/how-to-perform-a-logistic-regression-in-r/
  # dataset$X2.1 = ifelse(dataset$X2.1==4, 1, 0)
  training = dataset[-fold,]
  testing = dataset[fold,]
  model4 <- glm(formula=training$type ~., family=binomial(link='logit'), data=training)
  model4.pred = predict(model4, testing, type = "response")
  model4.pred = ifelse(model4.pred<0.5, '0', '1')
  testing$type = ifelse(testing$type==0, '0', '1')
  model4.scores = score(testing$type, model4.pred)
  
  # Model 5: KNN (K = 3)
  training = dataset[-fold,]
  testing = dataset[fold,]
  model5.pred <- knn(train = training, test = testing, cl = training$type, k = 3)
  model5.pred = ifelse(model5.pred == 0, '0', '1')
  testing$type = ifelse(testing$type==0, '0', '1')
  model5.scores = score(testing$type, model5.pred)
  r <- rbind(model1.scores, model2.scores, model3.scores, model4.scores, model5.scores) 
  rownames(r)<- c('NaiveBayes', 'TANBayes', 'LDA', 'LR', 'KNN') 
  return(r)
}



nFolds <- 10
folds <- createFolds(dataset$type, k = nFolds) 
cv<-lapply(folds, trainAndTestFold, dataset=dataset)
meanCV10 <- Reduce('+',cv) / nFolds
write.table(meanCV10,"meanCV_wineCat10.txt",sep="\t",row.names=TRUE)
stdCV10 <- sqrt(Reduce('+',lapply(cv,function(x) (x - meanCV10)^2)))
write.table(stdCV10,"stdCV_wineCat10.txt",sep="\t",row.names=TRUE)


nFolds <- 20
folds <- createFolds(dataset$type, k = nFolds) 
cv<-lapply(folds, trainAndTestFold, dataset=dataset)
meanCV20 <- Reduce('+',cv) / nFolds
write.table(meanCV20,"meanCV_wineCat20.txt",sep="\t",row.names=TRUE)
stdCV20 <- sqrt(Reduce('+',lapply(cv,function(x) (x - meanCV20)^2)))
write.table(stdCV20,"stdCV_wineCat20.txt",sep="\t",row.names=TRUE)
