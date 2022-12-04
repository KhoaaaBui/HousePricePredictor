import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas

def train_data():
    # parsing the data
    train_df = pandas.read_csv('train.csv', delimiter=',')

    # clean the data
    train_set = train_df.drop(['Id', 'SalePrice'], axis=1)
    train_target = train_df['SalePrice']
    train_set = pandas.get_dummies(train_set)
    na_vals = []

    # find columns containing null values
    for cols in train_set.columns:
        if (train_set[cols].isnull().sum() > 0):
            na_vals.append(cols)
    for cols in na_vals:
        if(cols == 'GarageYrBlt'):
            train_set[cols].fillna(train_set[cols].mean(numeric_only=True), inplace=True)
        else:
            train_set[cols].fillna(0, inplace=True)

    # splitting the data
    X_train, X_test, y_train, y_test = train_test_split(train_set, train_target, test_size = 0.2)
    reg = linear_model.LinearRegression()
    model = reg.fit(X_train, y_train)
    score = reg.score(X_test, y_test)
    print("Test accuracy: ", score)

    print(model.coef_)
    print(model.summary())
    #plot.scatter(train_set, train_target, color="black")
    #plot.plot(train_set, train_target, color="blue", linewidth=3)

if __name__ == '__main__':
    train_data()