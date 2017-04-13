library(RTextTools)
library(class)
setwd("C:/Users/Biligiri Vasan/Desktop/Spring 2016/ML/Assignment 4")
r<-read_data("train",type = "folder",index = "labels.csv", warn=F)
doc_matrix <- create_matrix(r$Text.Data, language="english", removeNumbers=TRUE, stemWords=TRUE, removeSparseTerms=.998)
container <- create_container(doc_matrix, r$Labels, trainSize=1:2966, testSize=2967:4942, virgin=FALSE)

#train models
SVM <- train_model(container,"SVM")
GLMNET <- train_model(container,"GLMNET")
MAXENT <- train_model(container,"MAXENT")
BOOSTING <- train_model(container,"BOOSTING")
RF <- train_model(container,"RF")
TREE <- train_model(container,"TREE")
KNN <- knn(train,test,k=15,prob = FALSE)

#test models
SVM_CLASSIFY <- classify_model(container, SVM)
GLMNET_CLASSIFY <- classify_model(container, GLMNET)
MAXENT_CLASSIFY <- classify_model(container, MAXENT)
BOOSTING_CLASSIFY <- classify_model(container, BOOSTING)
RF_CLASSIFY <- classify_model(container, RF)
TREE_CLASSIFY <- classify_model(container, TREE)

#analytics for each classifier
analytics_svm <- create_analytics(container, cbind(SVM_CLASSIFY))
analytics_glmnet <- create_analytics(container, cbind(GLMNET_CLASSIFY))
analytics_maxent <- create_analytics(container, cbind(MAXENT_CLASSIFY))
analytics_boosting <- create_analytics(container, cbind(BOOSTING_CLASSIFY))
analytics_rf <- create_analytics(container, cbind(RF_CLASSIFY))
analytics_tree <- create_analytics(container, cbind(TREE_CLASSIFY))

#accuracy
analytics_svm@label_summary
analytics_glmnet@label_summary
analytics_maxent@label_summary
analytics_boosting@label_summary
analytics_rf@label_summary
analytics_tree@label_summary

#recall,f-score and precision
analytics <- create_analytics(container, cbind(SVM_CLASSIFY, BOOSTING_CLASSIFY, BAGGING_CLASSIFY, RF_CLASSIFY, GLMNET_CLASSIFY, TREE_CLASSIFY, MAXENT_CLASSIFY))
summary(analytics)
