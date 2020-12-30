import argparse
import yaml
from src.data_load import load_data
from src.feature import *
from src.data_split import data_split
from src.train import *
from src.predict import *


def evaluate(config_path):
    config = yaml.safe_load(open(config_path))

    test_dir = config['test_set_dir']
    train_dir = config['train_set_dir']
    model_save_dir = config['model_save_dir']
    random_state = config['random_state']
    train, test = load_data(train_dir, test_dir)

    train = preprocess_feature(train)
    test = preprocess_feature(test)

    train, test = fare_feature(train, test)

    train, test = age_feature(train, test)

    train, test = sex_embarked_feature(train, test)

    train = drop_feature(train)
    test = drop_feature(test)

    x_train, x_val, y_train, y_val = data_split(train, random_state)

    gbk = train_model(x_train, y_train)

    prediction = predict(gbk, x_val)

    score = score_acc(prediction, y_val)
    print(f"acc_score: {score}")

    save_predict(gbk, test)

    save_model(gbk, model_save_dir)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config_path', dest='config_path', required=True)
    args = args_parser.parse_args()

    evaluate(config_path=args.config_path)
