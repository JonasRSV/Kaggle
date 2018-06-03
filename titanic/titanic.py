import numpy   as np
import pandas  as pd
import seaborn as sb
import matplotlib.pyplot as plt
import time

kaggle = "~/.kaggle/competitions/titanic/"

train = pd.read_csv(kaggle + "train.csv")
test  = pd.read_csv(kaggle + "test.csv")


print("Train shape {}, Test shape {}\n".format(train.shape, test.shape))



print("*** TRAIN ***")
train_stats = train.describe()
print(train_stats)

print("\n*** TEST ***")
test_stats  = test.describe()
print(test_stats)


print(train.isnull().sum())

"""For this problem i'd like to make the categorical data into numerical data."""
print(train.dtypes)

train["Cabin"].fillna("", inplace=True)
train["Cabin"] = train["Cabin"].map(lambda x: len(x.split()))

test["Cabin"].fillna("", inplace=True)
test["Cabin"] = test["Cabin"].map(lambda x: len(x.split()))


"""
Most cabins are NaN, but some information like number of cabins gives a nice entrophy gain...
Nan is set to being 0.
"""

train["Name"] = train["Name"].map(len)
train["Sex"]  = train["Sex"].map(lambda x: 0 if x == "male" else 1)

test["Name"] = test["Name"].map(len)
test["Sex"]  = test["Sex"].map(lambda x: 0 if x == "male" else 1)

def age_dist(x):
    """ Age groups are more descriptive than actual age. """
    if x < 10:
        return 0 
    elif x < 20:
        return 1
    elif x < 40:
        return 2
    elif x < 60:
        return 3
    else:
        return 4


train["Age"].fillna(train_stats["Age"]["mean"], inplace=True)
train["Age"]  = train["Age"].map(age_dist)

test["Age"].fillna(train_stats["Age"]["mean"], inplace=True)
test["Age"]  = test["Age"].map(age_dist)



train["Relatives"] = train["SibSp"] + train["Parch"]

test["Relatives"] = test["SibSp"] + test["Parch"]




"""Ticket number is unique per passenger aswell as Passenger Id so we'll drop these."""
train.drop(columns=["Ticket", "PassengerId"], inplace=True)

test.drop(columns=["Ticket", "PassengerId"], inplace=True)



def embark_dist(x):
    if x   == "C":
        return 0
    elif x == "S":
        return 1
    elif x == "Q":
        return 2
    else:
        return 3

train["Embarked"] = train["Embarked"].map(embark_dist)
test["Embarked"] = test["Embarked"].map(embark_dist)


from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import model_selection

NB = GaussianNB()
RF = RandomForestClassifier(30)
BD = AdaBoostClassifier()

train, validate = model_selection.train_test_split(train, test_size=0.25)

classifiers = {
                "Naive Bayes": NB
              , "Random Forest": RF
              , "Ada Boosted Desicion Tree": BD
             }


train_features = train.drop(columns=["Survived"])
train_target   = train["Survived"]

validate_features  = validate.drop(columns=["Survived"])
validate_target    = validate["Survived"]



for name, classifier in classifiers.items():
    print("Training: {}".format(name))
    timestamp = time.time()

    classifier.fit(train_features, train_target)

    training_took = time.time() - timestamp
    predictions   = classifier.predict(validate_features)
    correct = np.sum(predictions == validate_target)

    print("Training took {}, Accuracy: {}".format(training_took, correct / len(validate_features)))






