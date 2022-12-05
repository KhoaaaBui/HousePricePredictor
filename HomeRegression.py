import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sb
import pandas

def train_data():
    # parsing the data
    train_df = pandas.read_csv('train.csv', delimiter=',')

    # visualize the correlation of features with house price
    price_corr = train_df.corr()['SalePrice'].sort_values()[:-1]
    fig, ax = plot.subplots(figsize = (12, 2), dpi = 100)
    sb.set_theme()
    ax = sb.barplot(x=price_corr.index, y=price_corr.values)
    plot.title("Correlation of Features with Sale Price")
    plot.ylabel("Correlation")
    plot.xticks(rotation = 90, fontsize = 6)
    # plot.show()

    # Price distribution
    plot.figure(figsize = (7, 8))
    sb.distplot(train_df['SalePrice'])
    plot.title("House Price Distribution")
    # plot.show()

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
    y_pred = model.predict(X_test)
    score = reg.score(X_test, y_test)

    # plot prediction scatter plot
    plot.figure(figsize=(12,12))
    plot.scatter(y_test, y_pred, c='red')
    # rescaling the limit
    p1 = max(max(y_pred), max(y_test))
    p2 = min(min(y_pred), min(y_test))
    plot.plot([p1, p2], [p1, p2], 'b-')
    plot.axis('equal')
    plot.xlabel('Result', fontsize=12)
    plot.ylabel('Predictions', fontsize=12)
    plot.title("Predictions vs Result")
    plot.show()

    print("Test accuracy: ", score)

    #print(model.coef_)
    #plot.scatter(train_set, train_target, color="black")
    #plot.plot(train_set, train_target, color="blue", linewidth=3)

if __name__ == '__main__':
    train_data()