# coding: utf-8

# Date __27 January 2019__
# Author __Niraj Dev Pandey__


import pickle
import pandas as pd
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def library_check():
    import numpy as np
    if np.__version__ != '1.15.4':
        print("The project is developed on NumPy 1.15.4")
        print("you are running on numpy {} version".format(np.__version__))
    import pandas as pd
    if pd.__version__ != '0.23.4':
        print("The project is developed on Pandas 0.23.4")
        print("you are running on Panda {} version".format(pd.__version__))
    import sklearn
    if sklearn.__version__ != '0.19.2':
        print("The project is developed on Sklearn 0.19.2")
        print("you are running on Sklearn {} version".format(sklearn.__version__))
    else:
        print("congratulations...! you already have all the correct dependencies installed")


library_check()


def rename_columns(DataFrame):
    """
    Change columns names in the data-set for more clearity
    """
    DataFrame.rename(columns={'instant': 'id',
                              'dteday': 'datetime',
                              'weathersit': 'weather',
                              'hum': 'humidity',
                              'mnth': 'month',
                              'cnt': 'count',
                              'hr': 'hour',
                              'yr': 'year'}, inplace=True)


def process_datetime(datetime_columns):
    """
    Change Datetime to date and time values, set those values as columns, and
    set original datetime column as index
    """
    datetime_columns['datetime'] = datetime_columns['datetime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    datetime_columns['year'] = datetime_columns.datetime.apply(lambda x: x.year)
    datetime_columns['month'] = datetime_columns.datetime.apply(lambda x: x.month)
    datetime_columns['day'] = datetime_columns.datetime.apply(lambda x: x.day)
    datetime_columns['hour'] = datetime_columns.datetime.apply(lambda x: x.hour)
    datetime_columns['weekday'] = datetime_columns.datetime.apply(lambda x: x.weekday())
    datetime_columns.set_index('datetime', inplace=True)
    return datetime_columns


def drop_useless_features(DataFrame, features):
    """
    Drop specified list of features
    """
    DataFrame.drop(features, inplace=True, axis=1)
    if features is None:
        raise FeatureNotProvided('Please provide the list of feature which you want to drop')
    return DataFrame


def one_hot_encoding(DataFrame, categorical_features):
    """
    Takes list of categorical features and turns them into One-Hot Encoding
    """
    DataFrame = pd.get_dummies(DataFrame, columns=categorical_features, drop_first=True)
    return DataFrame


def split_data(preprocessed_data):
    """
    This function will divide the data into train ans test set
    """
    x_train, x_test, y_train, y_test = train_test_split(
        preprocessed_data.drop('count', axis=1),
        preprocessed_data['count'],
        test_size=0.2,
        random_state=42)
    return x_train, x_test, y_train, y_test


def check_input_shape(x_train, y_train):
    """
    This fuction will return 'Bad input shape' in case data is splitted wrongly
    X normaly contain 2D shape while Y has 1D array shape
    """
    sklearn.utils.check_X_y(x_train, y_train,
                            accept_sparse=False,
                            dtype='numeric', order=None,
                            copy=False, force_all_finite=True,
                            ensure_2d=True, allow_nd=False,
                            multi_output=False, 
                            ensure_min_samples=1, 
                            ensure_min_features=1,
                            y_numeric=False, 
                            warn_on_dtype=False,
                            estimator=None)
    

def train_model(x_train, y_train):
    """
    Train the model on traning set
    """
    reg = RandomForestRegressor()
    reg.fit(x_train, y_train)
    filename = 'forest_model.sav'
    pickle.dump(reg, open(filename, 'wb'))


def test_model(saved_model, X_test):
    loaded_model = pickle.load(open(saved_model, 'rb'))
    try:
        loaded_model
    except:
        raise TrainModelYourself("The loaded model is not found, please train your model by using 'Fit' function")
    result = loaded_model.predict(X_test)
    return result


def model_evaluation(y_pred, y_true):
    """
    Evaluate the performance of the model. This will give us MSLE error on test set
    """
    error = mean_squared_log_error(y_pred, y_true)
    return error


def main():
    try:
        data = pd.read_csv('./Bike-Sharing-Dataset/hour.csv')
    except:
        raise FileNotFoundError('Please enter the correct file path of Bike-Sharing dataset')
    try:
        saved_model = 'forest_model.sav'
    except:
        raise FileNotFoundError("[Errno 2] No such file or directory found:", saved_model)

    rename_columns(data)
    data = drop_useless_features(data, ['registered', 'atemp'])
    data = process_datetime(data)
    data = one_hot_encoding(data, ['season', 'holiday', 'workingday', 'weather', 'month', 'hour', 'weekday'])
    x_train, x_test, y_train, y_test = split_data(data)
    check_input_shape(x_train, y_train)
    train = train_model(x_train, y_train)
    test = test_model(saved_model, x_test)
    error = model_evaluation(test, y_test)
    print('Here are top 100 prediction by our model')
    print()
    print(test[:100])
    print()
    print('The mean square log error is:', error)


if __name__ == "__main__":
    main()
