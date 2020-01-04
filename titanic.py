# linear algebra
import numpy as np 

# data processing
import pandas as pd 


from matplotlib import pyplot as plt
from matplotlib import style

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing


#read data
test= pd.read_csv("test.csv")
train= pd.read_csv("train.csv")



train_test_data = [train, test] # combining train and test dataset

#extract title from name
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
#print(train['Title'].value_counts())

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, 
                 "Master": 4, "Dr": 0, "Rev": 0, "Col": 0, "Major": 0, "Mlle": 0,"Countess": 0,
                 "Ms": 0, "Lady": 0, "Jonkheer": 0, "Don": 0, "Dona" : 0, "Mme": 0,"Capt": 3,"Sir": 3 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# delete unnecessary feature from dataset
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)
train.drop('Cabin',axis=1,inplace=True)
test.drop('Cabin',axis=1,inplace=True)
train.drop('Ticket',axis=1,inplace=True)
train.drop('PassengerId',axis=1,inplace=True)
test.drop('Ticket',axis=1,inplace=True)
print(train.head())



sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


# fill missing age with mean age for each title (Mr, Mrs, Miss, Master, Others)
train["Age"].fillna(train.groupby("Title")["Age"].transform("mean"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("mean"), inplace=True)

#train["Age"] = preprocessing.normalize(train["Age"])

#normalizing age
train["Age"]=((train["Age"]-train["Age"].min())/(train["Age"].max()-train["Age"].min()))
test["Age"]=((test["Age"]-train["Age"].min())/(train["Age"].max()-train["Age"].min()))

#fill embarked missing values with most frequent
train["Embarked"].fillna("S", inplace=True)
test["Embarked"].fillna("S", inplace=True)

E_mapping={"S":0,"C":1,"Q":2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(E_mapping)

#filling missing values and normalizing
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("mean"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("mean"), inplace=True)
train["Fare"]=((train["Fare"]-train["Fare"].min())/(train["Fare"].max()-train["Fare"].min()))
test["Fare"]=((test["Fare"]-train["Fare"].min())/(train["Fare"].max()-train["Fare"].min()))

print(train.head())
train.info()
print(train['Embarked'].value_counts())
test.info()


train_data = train.drop('Survived', axis=1)
target = train['Survived']

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

#testing different algorithms
clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
#print(score)
print("KNN", round(np.mean(score)*100, 2))

clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
#print(score)
print("DTC", round(np.mean(score)*100,2))


clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
#print(score)
print("RFC", round(np.mean(score)*100,2))


clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
#print(score)
print("GNB", round(np.mean(score)*100,2))


clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
#print(score)
print("SVC", round(np.mean(score)*100,2))


"""clf = Perceptron()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
#print(score)
print("Perception",round(np.mean(score)*100,2))"""


clf = LogisticRegression()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
#print(score)
print("LR",round(np.mean(score)*100,2))

#prediction
clf = LogisticRegression()
clf.fit(train_data, target)

test_data = test.drop("PassengerId", axis=1).copy()
prediction = clf.predict(test_data)


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })

submission.to_csv('submission.csv', index=False)

print(prediction)

