library(caret)
library(e1071)
library(randomForest)
library(xgboost)
library(FeatureHashing)
library(Matrix)
library(tm)
library(plyr)

rm(list=ls())
###data loading
train <- read.csv('train.csv')
test <- read.csv("test.csv")
combined <- as.factor(c(as.character(train$tweet) ,as.character(test$tweet)))
# str(combined)
# combined
# str(train)
# str(test)

###word processing
corpus <- Corpus(VectorSource(combined))
corpus <- tm_map(corpus, content_transformer(tolower))  
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stemDocument)
corpus
as.character(corpus[[1]])
dtm <- DocumentTermMatrix(corpus)
dtm <- removeSparseTerms(dtm, 0.999)
dim(dtm)
dtm

convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}

datasetNB <- apply(dtm, 2, convert_count)
dataset <- as.data.frame(as.matrix(datasetNB))
dim(dataset)
train_set<-dataset[1:22500,]
train_set$Class=as.factor(train$sentiment)
test_set<-dataset[22501:30000,]
test_set$Class <- 0
set.seed(123)
split = sample(2,nrow(train_set),prob = c(0.8,0.2),replace = TRUE)
train2_set = train_set[split == 1,]
valid_set = train_set[split == 2,]

ctrl <- trainControl(method="repeatedcv", number=10, repeats=3,search="grid",allowParallel = T)
classifier_nb <- naiveBayes(train_set[,-ncol(train_set)], train_set$Class, laplace = 1,trControl = ctrl,tuneLength = 3)
nb_pred <- predict(classifier_nb, type = 'class', newdata = test_set)
#confusionMatrix(table(nb_pred,valid_set$Class))

#tune for best mtry
#bestmtry <- tuneRF(train_set[,-ncol(train_set)], train_set$Class, stepFactor=0.1, improve=1e-5, ntree=500,trace=T,plot=T,mtryStart = 15)
#plot(bestmtry)
##best = 15
#model
rf_classifier = randomForest(x = train_set[,-ncol(train_set)], y=train_set$Class,data = train_set, ntrees = 500,mtry=15)
rf_pred <- predict(rf_classifier, newdata = test_set[,-ncol(test_set)])
#confusionMatrix(table(rf_pred,valid_set$Class))

svm_classifier_sigmoid <- svm(Class~., data=train2_set,kernel='sigmoid',gamma=0.5)
svm_classifier_poly <- svm(Class~., data=train2_set,kernel='poly',gamma=0.5)
svm_classifier_radial <- svm(Class~., data=train2_set,kernel='radial',gamma=0.5)
svm_classifier_linear <- svm(Class~., data=train2_set,kernel='linear')
svm_pred_sigmoid <- predict(svm_classifier_sigmoid,valid_set[,-ncol(valid_set)])
svm_pred_linear <- predict(svm_classifier_linear,valid_set[,-ncol(valid_set)])
svm_pred_poly <- predict(svm_classifier_poly,valid_set[,-ncol(valid_set)])
svm_pred_radial <- predict(svm_classifier_radial,valid_set)
confusionMatrix(table(svm_pred_sigmoid,valid_set$Class))
confusionMatrix(table(svm_pred_poly,valid_set$Class))
confusionMatrix(table(svm_pred_radial,valid_set$Class))
confusionMatrix(table(svm_pred_linear,valid_set$Class))


svm_classifier <- svm(Class~., data=train_set,kernel='linear',cost=0.175)
svm_pred <- predict(svm_classifier,test_set[,-ncol(test_set)])

xgbres<-read.csv('xgb_results.csv')
svm_output <- cbind.data.frame(test,sentiment=svm_pred)
nb_output <- cbind.data.frame(test,sentiment=nb_pred)
rf_output <- cbind.data.frame(test,sentiment=rf_pred)
svm_output_wo_tweets <- svm_output[c('Id' , "sentiment")]
rf_output_wo_tweets <- rf_output[c('Id' , "sentiment")]
nb_output_wo_tweets <- nb_output[c('Id' , "sentiment")]
rownames(svm_output_wo_tweets)<- NULL
rownames(rf_output_wo_tweets)<- NULL
rownames(nb_output_wo_tweets)<- NULL
xgb_svm<-merge(svm_output_wo_tweets,xgbres,by="Id")
xgb_svm$check<-ifelse(xgb_svm$sentiment.x==xgb_svm$sentiment.y,0,1)
svm_diff<-sum(xgb_svm$check)/nrow(xgb_svm)
xgb_rf<-merge(rf_output_wo_tweets,xgbres,by="Id")
xgb_rf$check<-ifelse(xgb_rf$sentiment.x==xgb_rf$sentiment.y,0,1)
rf_diff<-sum(xgb_rf$check)/nrow(xgb_rf)
xgb_nb<-merge(nb_output_wo_tweets,xgbres,by="Id")
xgb_nb$check<-ifelse(xgb_nb$sentiment.x==xgb_nb$sentiment.y,0,1)
nb_diff<-sum(xgb_nb$check)/nrow(xgb_nb)
comparison_all<-data.frame(cbind(svm_diff,rf_diff,nb_diff),row.names = "Model Difference")
colnames(comparison_all)<-c("SVM","Random Forest","Naive Bayes")
comparison_all

write.csv(svm_output_wo_tweets,"svm_results.csv", row.names = FALSE)