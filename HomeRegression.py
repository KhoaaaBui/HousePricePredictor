import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api
import matplotlib.pyplot as plot
import pandas

def train_data():
    train_df = pandas.read_table('train.csv', delimiter=',')
    train_set = train_df.drop(['Id', 'SalePrice'], axis=1)
    train_target = train_df['SalePrice'].values
    train_set = pandas.get_dummies(train_set)
    na_vals = []
    for cols in train_set.columns:
        if (train_set[cols].isnull().sum() > 0):
            na_vals.append(cols)
    for cols in na_vals:
        if(cols == 'GarageYrBlt'):
            train_set[cols].fillna(train_set[cols].mean(numeric_only=True), inplace=True)
        else:
            train_set[cols].fillna(0, inplace=True)
    reg = linear_model.LinearRegression()
    model = reg.fit(train_set, train_target)
    print(model.coef_)
    print(model.summary())
    #plot.scatter(train_set, train_target, color="black")
    #plot.plot(train_set, train_target, color="blue", linewidth=3)

if __name__ == '__main__':
    train_data()