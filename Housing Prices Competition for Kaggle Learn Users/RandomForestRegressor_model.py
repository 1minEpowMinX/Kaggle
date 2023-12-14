import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from os import path
import matplotlib.pyplot as plt

current_dir = path.dirname(__file__)
# * Path of the file to read
iowa_file_path = path.join(current_dir, 'train.csv')

# * Load the data, and separate the target
home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice


def get_features() -> list:
    with open(path.join(current_dir, 'main_features_house.txt')) as info:
        features = [row.split(':')[0] for row in info.readlines()]
        return features


# * Create X
features = get_features()  # TODO: improve features collection

# * Select columns corresponding to features, and preview the data
X = home_data[features]
# print(X.head())

# * Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


def get_model_importance(model):
    importances = model.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.barh(get_features(), importances)
    plt.xlabel('Важность признака')
    plt.ylabel('Признаки')
    plt.title('Важность признаков')
    return plt.show()


def get_mae(n_estimators=100, max_depth=None, max_leaf_nodes=None, importance=False) -> str:
    # * Define a random forest model
    rf_model = RandomForestRegressor(
        random_state=1, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, n_estimators=n_estimators)
    rf_model.fit(train_X, train_y)
    rf_val_predictions = rf_model.predict(val_X)
    rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
    if importance:
        return get_model_importance(rf_model)
    return f"Validation MAE for Random Forest Model: {rf_val_mae:,.0f}"


print(get_mae(n_estimators=100, max_depth=15,
      max_leaf_nodes=None, importance=False))

rf_model_on_full_data = RandomForestRegressor(
    random_state=1, max_depth=20, max_leaf_nodes=1000)
rf_model_on_full_data.fit(X, y)

test_data_path = path.join(current_dir, 'test.csv')
test_data = pd.read_csv(test_data_path)

features = get_features()
test_X = test_data[features]

# * Фильтруем столбцы с отсутствующими значениями (NaN)
# columns_with_nan = test_X.columns[test_X.isna().any()].tolist()

# * Выводим только столбцы с отсутствующими значениями
# df_with_nan = test_X[columns_with_nan]
# print(df_with_nan)

# test_preds = rf_model_on_full_data.predict(test_X)

# output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds})
# output.to_csv(path.join(current_dir, 'submission.csv'), index=False)
