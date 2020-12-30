from sklearn.metrics import accuracy_score
import pandas as pd
import pickle


def predict(gbk, x_val):
    return gbk.predict(x_val)


def score_acc(y_pred, y_val):
    return round(accuracy_score(y_pred, y_val) * 100, 2)


def save_predict(gbk, test):
    # set ids as PassengerId and predict survival
    ids = test['PassengerId']
    predictions = gbk.predict(test.drop('PassengerId', axis=1))

    # set the output as a dataframe and convert to csv file named submission.csv
    output = pd.DataFrame({'PassengerId': ids, 'Survived': predictions})
    output.to_csv('data/submission.csv', index=False)


def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)
