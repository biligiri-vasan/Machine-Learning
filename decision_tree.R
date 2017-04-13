library(party)
#library(tree)
index <- sample(1:nrow(adult),round(0.80*nrow(adult)))
train <- adult[index,]
test <- adult[-index,]
adult_ctree <- ctree(train$V15 ~ .,data=train)
print(adult_ctree)
a = predict(adult_ctree,test)
plot(adult_ctree)

index <- sample(1:nrow(wine),round(0.80*nrow(wine)))
train <- wine[index,]
test <- wine[-index,]
wine_ctree <- ctree(wine$V1 ~ ., data = wine)
plot(wine_ctree)
a = predict(wine_ctree,test)
car_ctree <- ctree(car$V7 ~ .,data=car)
plot(car_ctree)

transfusion_ctree <- ctree(transfusion$V5 ~ ., data=transfusion)
plot(transfusion_ctree)

house.votes.84_ctree <- ctree(house.votes.84$republican ~ ., data= house.votes.84)
plot(house.votes.84_ctree)

library(neuralnet)
nn <- neuralnet(case~age+parity+induced+spontaneous, data=infert, hidden=0, err.fct="ce", linear.output=FALSE)