require(ada)
require(class)
require(adabag)
require(randomForest)
require(rpart)
require(ipred)
require(maboost)
require(gbm)

setwd("C:/Users/Biligiri Vasan/Desktop/Spring 2016/ML/Assignment 5")

dataset_collection <- as.array(c('wdbc.data','wpbc.data', 'transfusion.data', 'soybean_small.data', 'ionosphere.data'))
class_array <- as.array(c(2,2,5,36,35))
header_array_check <- as.array(c('F','F','F','F','F'))

flag<-as.logical('F')

#reading and formatting data for input into classifiers
wdbc_data <- read.table("wdbc.data",header=flag,sep=",", na.strings = c("?"))
wdbc_data<- na.omit(wdbc_data)
wpbc_data <- read.table("wpbc.data",header=flag,sep=",", na.strings = c("?"))
wpbc_data<- na.omit(wpbc_data)
transfusionosphere_data <- read.table("transfusion.data",colClasses = c("numeric","numeric","numeric","numeric","factor"), header = TRUE, sep = ',')
soybean_data <- read.table("soybean_small.data",header = TRUE, sep = ',')
ionosphere_data <- read.table("ionosphere.data",header = TRUE, sep = ',')

data_list <- list()
data_list[[1]] <- wdbc_data
data_list[[2]] <- wpbc_data
data_list[[3]] <- transfusionosphere_data
data_list[[4]] <- soybean_data
data_list[[5]] <- ionosphere_data

for ( d in 1:length(dataset_collection)){
  data_Url<- dataset_collection[d]
  flag<-as.logical(header_array_check[d])
  cls<-as.integer(class_array[d])
  
  data <- data_list[[d]] 
  
  class_attr <- names(data[cls])
  col_names <- names(data[,-cls])
  
  accuracy <- list()
  accuracy[[1]] <- list()
  accuracy[[2]] <- list()
  accuracy[[3]] <- list()
  accuracy[[4]] <- list()
  accuracy[[5]] <- list()
  
  folds <- cut(seq(1,nrow(data)),breaks=10,labels=FALSE)
  
  cat("Dataset: ",dataset_collection[d],"\n")
  
  #Perform 10 fold cross validation
  for(i in 1:10){
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    test_data_set <- data[testIndexes, ]
    training_data_set <- data[-testIndexes, ]
    
    train_Class_data<-training_data_set[,cls]
    test_Class_data<-test_data_set[,cls]
    
    #KNN
    knn_model <- knn(training_data_set[,-cls], test_data_set[,-cls], train_Class_data, k = 9, prob=TRUE, use.all = FALSE)
    knn_accuracy <- mean(knn_model == test_Class_data)*100
    accuracy[[1]][[i]] <- knn_accuracy
    
    #AdaBoost
    if(dataset_collection[[d]]=="soybean_small.data" ) {
      adaBoost_model <- maboost(as.formula(paste(names(training_data_set[36]),sep="","~.")),data=training_data_set,iter =30 , type="discrete")
      adaBoost_predicted <- predict(adaBoost_model,test_data_set)
      ada_accuracy <- sum(test_Class_data==adaBoost_predicted)/length(adaBoost_predicted)*100
      accuracy[[2]][[i]] <- ada_accuracy
    } else {
      adaBoost_model <- ada(as.formula(paste(class_attr[1],sep="","~.")),data=training_data_set,iter =30 , type="discrete")
      adaBoost_predicted <- predict(adaBoost_model,test_data_set)
      ada_accuracy <- sum(test_Class_data==adaBoost_predicted)/length(adaBoost_predicted)*100
      accuracy[[2]][[i]] <- ada_accuracy
    }
    
    #Bagging
    tryCatch(
      {
        bagging_model <- bagging(as.formula(paste(class_attr[1],sep="","~.")),data=training_data_set, mfinal = 100)
        bagging_predicted <- predict(bagging_model, test_data_set)
        if(dataset_collection[[d]]=="soybean_small.data") {
          bagging_accuracy <- sum(as.character(bagging_predicted) == as.character(test_Class_data))/length(bagging_predicted)*100 
        } else {
          bagging_accuracy <- sum(bagging_predicted == test_Class_data)/length(bagging_predicted)*100  
        } 
        accuracy[[3]][[i]] <- bagging_accuracy
      },
      error=function(cond) {
      },
      warning=function(cond) {
      }
    )    
    
    #RF
    RF_model <- randomForest(as.formula(paste(class_attr[1],sep="","~.")),data=training_data_set, importance=TRUE)
    RF_predicted <- predict(RF_model, test_data_set, type="class")
    RF_accuracy <- mean(RF_predicted == test_Class_data)*100 
    accuracy[[4]][[i]] <- RF_accuracy
  }
  
  knn_accuracy <- mean(as.numeric(accuracy[[1]]))
  cat("Classifier = KNN",", accuracy= ",     ada_accuracy,"\n")
  
  ada_accuracy <- mean(as.numeric(accuracy[[2]]))
  cat("Classifier = AdaBoosing",", accuracy= ",     ada_accuracy,"\n")
  
  accuracy[[3]] = lapply(accuracy[[3]],function(x) if(is.null(x)) 0 else x) 
  bagging_accuracy <- mean(as.numeric(accuracy[[3]]))
  cat("Classifier = Bagging",", accuracy= ",     bagging_accuracy,"\n")
  
  RF_accuracy <- mean(as.numeric(accuracy[[4]]))
  cat("Classifier = RF",", accuracy= ",     RF_accuracy,"\n")  
  
  if(dataset_collection[[d]]=="soybean_small.data") {
    g <- gbm(as.formula(paste(class_attr[1],sep="","~.")),data=data, distribution = "multinomial" ,keep.data=TRUE, n.minobsinnode = 4, cv.folds = 9)
  }
  else {
    g <- gbm(as.formula(paste(class_attr[1],sep="","~.")),data=data, distribution = "multinomial" ,keep.data=TRUE, n.minobsinnode = 4, cv.folds = 10)  
  }
  g_predicted <- predict(g,n.trees = 10)
  gp <- colnames(g_predicted)[apply(g_predicted,1,function(i){which(i==max(i))})]
  g_accuracy <- mean(gp == data[cls])*100 
  cat("Classifier = Gradient Boosting",", accuracy= ", g_accuracy,"\n\n\n")
}