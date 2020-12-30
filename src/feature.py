import pandas as pd
import numpy as np

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']


def preprocess_feature(data):
    data["Age"] = data["Age"].fillna(-0.5)
    data['AgeGroup'] = pd.cut(data["Age"], bins, labels=labels)
    data["CabinBool"] = (data["Cabin"].notnull().astype('int'))
    return data.fillna({"Embarked": "S"})


def fare_feature(train, test):
    # fill in missing Fare value in test set based on mean fare for that Pclass
    for x in range(len(test["Fare"])):
        if pd.isnull(test["Fare"][x]):
            pclass = test["Pclass"][x]  # Pclass = 3
            test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)

    # map Fare values into groups of numerical values
    train['FareBand'] = pd.qcut(train['Fare'], 4, labels=[1, 2, 3, 4])
    test['FareBand'] = pd.qcut(test['Fare'], 4, labels=[1, 2, 3, 4])

    return train, test


def age_feature(train, test):
    # create a combined group of both datasets
    combine = [train, test]

    # extract a title for each Name in the train and test datasets
    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    # replace various titles with more common names
    for dataset in combine:
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
                                                     'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

        dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # map each of the title groups to a numerical value
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)

    age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}

    for x in range(len(train["AgeGroup"])):
        if train["AgeGroup"][x] == "Unknown":
            train["AgeGroup"][x] = age_title_mapping[train["Title"][x]]

    for x in range(len(test["AgeGroup"])):
        if test["AgeGroup"][x] == "Unknown":
            test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]

    # map each Age value to a numerical value
    age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
    train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
    test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

    return train, test


def sex_embarked_feature(train, test):
    sex_mapping = {"male": 0, "female": 1}
    train['Sex'] = train['Sex'].map(sex_mapping)
    test['Sex'] = test['Sex'].map(sex_mapping)
    embarked_mapping = {"S": 1, "C": 2, "Q": 3}
    train['Embarked'] = train['Embarked'].map(embarked_mapping)
    test['Embarked'] = test['Embarked'].map(embarked_mapping)
    return train, test


def drop_feature(data):
    data = data.drop(['Cabin'], axis=1)
    data = data.drop(['Ticket'], axis=1)
    data = data.drop(['Name'], axis=1)
    data = data.drop(['Age'], axis=1)
    data = data.drop(['Fare'], axis=1)
    return data
