import numpy as np
import pandas as pd
from os import path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

current_dit = path.dirname(__file__)

df_train = pd.read_csv(path.join(current_dit, 'train.csv'))
df_test = pd.read_csv(path.join(current_dit, 'test.csv'))


def get_review_data():
    # * Просматриваем данные
    print(df_train.head())
    print(df_train.shape)
    print(df_train['SalePrice'].describe())


# get_review_data()


def get_gist():
    # * гистограмма данных
    f, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df_train['SalePrice'])
    return plt.show()


# get_gist()


def get_knew_and_kurt():
    # * Рассчитываем асимметрию и эксцесс
    print("Ассиметрия: %f" % df_train['SalePrice'].skew())
    print("Эксцесс: %f" % df_train['SalePrice'].kurt())


# get_knew_and_kurt()


def get_distrib_plt():
    # * распределение на необработанных данных
    fig = plt.figure(figsize=(14, 8))
    fig.add_subplot(1, 2, 1)
    res = stats.probplot(df_train['SalePrice'], plot=plt)

    # * Распределение при условии, что мы прологарифмировали 'SalePrice'
    fig.add_subplot(1, 2, 2)
    res = stats.probplot(np.log1p(df_train['SalePrice']), plot=plt)
    plt.show()


# * логарифмирование тренировочной выборки
df_train['SalePrice'] = np.log1p(df_train['SalePrice'])


def get_corr():
    # * Матрица корреляции
    corrmat = df_train.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    k = 10  # количество коррелирующих признаков, которое мы хотим увидеть
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(train_df[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True,
                     fmt='.2f', annot_kws={'size': 10},
                     yticklabels=cols.values, xticklabels=cols.values)
    plt.show()


get_corr()
