from sklearn.ensemble import GradientBoostingClassifier


def train_model(x_train, y_train):
    gbk = GradientBoostingClassifier()
    gbk.fit(x_train, y_train)
    return gbk