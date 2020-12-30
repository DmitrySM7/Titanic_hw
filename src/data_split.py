from sklearn.model_selection import train_test_split


def data_split(train, random_state):
    predictors = train.drop(['Survived', 'PassengerId'], axis=1)
    target = train["Survived"]
    return train_test_split(predictors, target, test_size=0.22, random_state=random_state)
