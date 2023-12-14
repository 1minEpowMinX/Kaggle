import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from os import path

current_dir = path.dirname(__file__)
# * Path of the file to read
iowa_data_path = path.join(current_dir, 'train.csv')

# * Iowa home data for predicting value
home_data = pd.read_csv(iowa_data_path)
feature_names = ["LotArea", "YearBuilt", "1stFlrSF",
                 "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
# * X - features, y - prediction target
X, y = home_data[feature_names], home_data.SalePrice
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


def review_data(frame: pd.DataFrame) -> None:
    '''
    Use this function to check data for errors.

    :param: frame
    :type: pd.DataFrame
    :return: None
    :rtype: None
    '''
    print(frame.describe(), frame.head(), sep='\n\n')


def get_mae(max_leaf_nodes: int, train_X: pd.Series, val_X: pd.Series, train_y: pd.Series, val_y: pd.Series) -> float:
    '''
    Return MAE of the training data.

    :param: max_leaf_nodes
    :type: int
    :param: train_X
    :type: pd.Series
    :param: val_X
    :type: pd.Series
    :param: train_y
    :type: pd.Series
    :param: val_y
    :type: pd.Series
    :return: MAE
    :rtype: float
    '''
    # * Specify Model
    iowa_model = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes, random_state=1)
    # * Fit Model with the training data
    iowa_model.fit(train_X, train_y)
    # * Make validation predictions and calculate mean absolute error
    val_predictions = iowa_model.predict(val_X)
    mae = mean_absolute_error(val_y, val_predictions)
    return mae


def get_candidate_max_leaf_nodes(candidate_max_leaf_nodes: list) -> str:
    '''
    Returns the maximum number of max leaf nodes according to input data.

    :param: candidate_max_leaf_nodes.
    :type: list
    :return: best max leaf node
    :rtype: str
    '''
    scores = {node: get_mae(node, train_X, val_X, train_y, val_y)
              for node in candidate_max_leaf_nodes}
    return f"Best max leaf node: {min(scores, key=scores.get):,.0f}"


def main() -> None:
    print(get_candidate_max_leaf_nodes([5, 25, 50, 100, 250, 500]))
    #  * make optimal leaf node size according to scores
    final_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
    final_model.fit(X, y)  # * fit the final model
    predictions = final_model.predict(X)
    print('Price prediction:', [round(predict) for predict in predictions[:5]])
    print('Actual target values for house price:', y.head().tolist())


if __name__ == '__main__':
    main()
