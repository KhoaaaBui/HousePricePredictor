import sklearn
import category_encoders
import csv
import pandas

def train_data():
    train_df = pandas.read_table('train.csv', delimiter=',')
    train_set = train_df.drop(['Id', 'SalePrice'], axis=1)
    train_target = train_df['SalePrice'].values
    train_set = pandas.get_dummies(train_set)
    print(train_set)


if __name__ == '__main__':
    train_data()