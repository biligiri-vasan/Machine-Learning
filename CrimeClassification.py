'''
SFO Crime Classification

CS 6375 - Machine Learning

Authors:
1. Asha Mary Thomas - axt143530
2. Biligiri Vasan - bxs152830
3. Somasundaram Ardhanareeswaran - sxa146230
'''

import pandas as pd

from sklearn import cross_validation
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans

from matplotlib import pyplot

trainLocation = "C:/train.csv"
testLocation = "C:/test.csv"


#Load and Format the data attributes
def dataMassaging(dataFrame):
    data = pd.DataFrame(index=range(len(dataFrame)))
    data = dataFrame.get(['X','Y'])
    dateTimeTemp = pd.to_datetime(dataFrame.Dates)
    day = dateTimeTemp.dt.day
    data['Day'] = day
    month = dateTimeTemp.dt.month
    data['Month'] = month
    year = dateTimeTemp.dt.year
    data['Year'] = year    
    hour = dateTimeTemp.dt.hour
    data['Hour'] = hour
    DayOfWeeks = dataFrame.DayOfWeek.unique()
    DayOfWeekMap = {}
    i = 0
    for x in DayOfWeeks:
        DayOfWeekMap[x] = i
        i = i + 1
    data = data.join(dataFrame['DayOfWeek'].map(DayOfWeekMap))
    PdDistricts = dataFrame.PdDistrict.unique()
    PdDistrictMap = {}
    i = 0
    for x in PdDistricts:
        PdDistrictMap[x] = i
        i = i + 1
    data = data.join(dataFrame['PdDistrict'].map(PdDistrictMap))     
    data['cluster_ids'] = dataFrame['cluster_ids']    
    data = data.drop('X',1)
    data = data.drop('Y',1)    
    features = data.values
    
    
    if 'Category' in dataFrame.columns:
        classes = dataFrame.Category.values    
    else:
        classes = None
        
    return features, classes

#Bar chart plot for Category of crime    
def visualizePrediction(predictions):
    data = pd.DataFrame(predictions)
    data['Category']=data[0]
    data.groupby('Category')['Category'].count().plot(kind='bar')
    return
    
#Plot the location coordinates    
def locationPlot(locations):
    pyplot.scatter(locations['X'], locations['Y'])
    return

#Read Data    
train = pd.read_csv(trainLocation)

#Filter outliers
train = train[train.Y < 38]

#convert to DataFrame
df = pd.DataFrame(train)

#Remove minority categories
df['category_count']  = df.groupby('Category')['Category'].transform('count')
df2 = df[df['category_count'].between(4280,174901)]
data_filtered = df2.drop('category_count',1)

#Form 1000 clusters for the X and Y coordinates
data = pd.DataFrame(data_filtered.get(['X','Y']))
locationPlot(data)
est = KMeans(n_clusters=1000, max_iter=100)
est.fit(data)

#Record cluster centers and their labels
cluster_centers = est.cluster_centers_
locations = pd.DataFrame(cluster_centers)
locations['X'] = locations[0]
locations['Y'] = locations[1]
locationPlot(locations)

#Add cluster IDs to DataFrame
data_filtered['cluster_ids']=est.labels_

#Data Massaging
features, classes = dataMassaging(data_filtered)

#####Classifiers to get Accuracy and Log loss functions#####

accuracies = {}
logLosses = {}

#RF Cross validation
print("RF Cross validation")
classifier = RandomForestClassifier()
totalScore = 0
totalLogLoss = 0
kfold = cross_validation.KFold(len(features),shuffle=True,n_folds=10)
for train, test in kfold:
    classifier.set_params(min_samples_split=1000)
    classifier.fit(features,classes)
    predictions = classifier.predict_proba(features[test])
    score = classifier.score(features[test], classes[test])
    log_loss = metrics.log_loss(classes[test], predictions)
    totalScore += score
    totalLogLoss += log_loss
    print("Accuracy : ", score)
    print("Log loss : ", log_loss)
totalScore = totalScore/10.0
totalLogLoss = totalLogLoss/10.0
print("Average Accuracy : ", totalScore)
print("Average Log Loss : ", totalLogLoss)
accuracies['RF'] = totalScore
logLosses['RF'] = totalLogLoss

#AdaBoost Cross Validation
print("Ada Cross validation")
classifier = AdaBoostClassifier(n_estimators=50)
totalScore = 0
totalLogLoss = 0
kfold = cross_validation.KFold(len(features),shuffle=True,n_folds=10)
for train, test in kfold:
    classifier.fit(features,classes)
    predictions = classifier.predict_proba(features[test])
    score = classifier.score(features[test], classes[test])
    log_loss = metrics.log_loss(classes[test], predictions)
    totalScore += score
    totalLogLoss += log_loss
    print("Accuracy : ", score)
    print("Log loss : ", log_loss)
totalScore = totalScore/10.0
totalLogLoss = totalLogLoss/10.0
print("Average Accuracy : ", totalScore)
print("Average Log Loss : ", totalLogLoss)
accuracies['Ada'] = totalScore
logLosses['Ada'] = totalLogLoss

#NB Cross Validation
print("NB Cross validation")
classifier = GaussianNB()
totalScore = 0
totalLogLoss = 0
kfold = cross_validation.KFold(len(features),shuffle=True,n_folds=10)
for train, test in kfold:
    classifier.fit(features,classes)
    predictions = classifier.predict_proba(features[test])
    score = classifier.score(features[test], classes[test])
    log_loss = metrics.log_loss(classes[test], predictions)
    totalScore += score
    totalLogLoss += log_loss
    print("Accuracy : ", score)
    print("Log loss : ", log_loss)
totalScore = totalScore/10.0
totalLogLoss = totalLogLoss/10.0
print("Average Accuracy : ", totalScore)
print("Average Log Loss : ", totalLogLoss)
accuracies['NB'] = totalScore
logLosses['NB'] = totalLogLoss

#predict category on testData
test = pd.read_csv(testLocation)

#predict clusters for test data
data2 = test.get(['X','Y'])
test_cluster_predict = est.predict(data2)
test['cluster_ids']=test_cluster_predict

test_features,_ = dataMassaging(test)

#RF predict
classifier1 = RandomForestClassifier()
classifier1.set_params(min_samples_split=1000)
classifier1.fit(features, classes)
predictions1 = classifier1.predict(test_features)
visualizePrediction(predictions1)

#AdaBoost predictx
classifier2 = AdaBoostClassifier(n_estimators=50)
classifier2.fit(features, classes)
predictions2 = classifier2.predict(test_features)
visualizePrediction(predictions2)

#NB predict
classifier3 = GaussianNB()
classifier3.fit(features, classes)
predictions3 = classifier3.predict(test_features)
visualizePrediction(predictions3)