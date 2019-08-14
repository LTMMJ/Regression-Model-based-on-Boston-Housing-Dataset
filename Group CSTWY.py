import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import boxcox1p
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

category=['MSZoning','Street','Alley','LotShape','LandContour','LotConfig','LandSlope','Neighborhood','Condition1','Condition2',
          'BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation',
          'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual',
          'Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType',
          'SaleCondition']

continuous= ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2','BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
             '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF','EnclosedPorch', '3SsnPorch',
             'ScreenPorch','PoolArea', 'MiscVal']

con_cate=['MSSubClass','OverallQual','OverallCond','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
          'TotRmsAbvGrd','Fireplaces','GarageCars','MoSold','YrSold']


def linear_regression(x_train, x_test, y_train, y_test):
    lg = LinearRegression()
    model=lg.fit(x_train, y_train)
    y_pred = lg.predict(x_test)
    mse=mean_squared_error(y_test, y_pred)
    #print("Linear: ", lg.get_params())	print parameters of linear regression
    print("MSE:",mse)
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()])
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title('Linear Regression')
    plt.show()
    print("Linear Regression Accuracy:", model.score(x_test, y_test))
    return y_pred

def kernel_ridge(x_train, x_test, y_train, y_test):
    krg = KernelRidge(alpha=2, kernel="linear", gamma=None, degree=3, coef0=1)
    model=krg.fit(x_train,y_train)
    y_pred=krg.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    #print("Kernel ridge: ", krg.get_params())	print parameters of kernel ridge regression
    print("MSE:", mse)
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title('Kernel Ridge Regression')
    plt.show()
    print("Kernel ridge Regression Accuracy:", model.score(x_test, y_test))
    return y_pred

def lasso_regression(x_train, x_test, y_train, y_test):
    ls = make_pipeline(RobustScaler(), Lasso(alpha=0.0007))
    model=ls.fit(x_train,y_train)
    y_pred=ls.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    #print("Lasso: ", ls.get_params())	print parameters of lasso regression
    print("MSE:", mse)
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title('Lasso Regression')
    plt.show()
    print("Lasso Regression Accuracy:", model.score(x_test, y_test))
    return y_pred

if __name__ == '__main__':
    train = pd.read_csv("train.csv")
    test = pd.read_csv('test.csv')
    train.drop("Id", axis=1, inplace=True)
    test.drop("Id", axis=1, inplace=True)

    train = train.drop(train[(train['LotFrontage'] > 300) & (train['BsmtFinSF1'] > 5000) & (train['TotalBsmtSF'] > 6000) & (train['1stFlrSF'] > 4000) & (train['GrLivArea'] > 4000)].index)
    train["SalePrice"] = np.log1p(train["SalePrice"])  ##log1p(x) := log(1+x)

    train_num = train.shape[0]
    test_num = test.shape[0]
    y = train.SalePrice.values

    all_data = pd.concat((train, test)).reset_index(drop=True)
    all_data.drop(['SalePrice'], axis=1, inplace=True)

    # missing data process
    # category feature of missing large data is replaced by NA
    all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
    all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
    all_data["Alley"] = all_data["Alley"].fillna("None")
    all_data["Fence"] = all_data["Fence"].fillna("None")
    all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

    # use neighborhood value to replace
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

    # category feature of missing little data is replace by NA, continuou feature of missing little data is replace by 0
    for col in ('GarageCond', 'GarageQual', 'GarageFinish', 'GarageType', 'BsmtCond', 'BsmtExposure', 'BsmtQual', 'BsmtFinType2','BsmtFinType1'):
        all_data[col] = all_data[col].fillna('None')
    all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna(0)

    # Utilities only have one values that is NoSewr
    all_data = all_data.drop(['Utilities'], axis=1)

    # the rate of missing low than 1% which missing data are replaced by the most values
    all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(all_data['MasVnrArea'].mode()[0])
    all_data['MasVnrType'] = all_data['MasVnrType'].fillna(all_data['MasVnrType'].mode()[0])
    all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
    all_data['BsmtHalfBath'] = all_data['BsmtHalfBath'].fillna(all_data['BsmtHalfBath'].mode()[0])
    all_data['Functional'] = all_data['Functional'].fillna(all_data['Functional'].mode()[0])
    all_data['BsmtFullBath'] = all_data['BsmtFullBath'].fillna(all_data['BsmtFullBath'].mode()[0])
    all_data['BsmtFinSF2'] = all_data['BsmtFinSF2'].fillna(all_data['BsmtFinSF2'].mode()[0])
    all_data['BsmtFinSF1'] = all_data['BsmtFinSF1'].fillna(all_data['BsmtFinSF1'].mode()[0])
    all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
    all_data['BsmtUnfSF'] = all_data['BsmtUnfSF'].fillna(all_data['BsmtUnfSF'].mode()[0])
    all_data['TotalBsmtSF'] = all_data['TotalBsmtSF'].fillna(all_data['TotalBsmtSF'].mode()[0])
    all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
    all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
    all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
    all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
    all_data['GarageArea'] = all_data['GarageArea'].fillna(all_data['GarageArea'].mode()[0])
    all_data['GarageCars'] = all_data['GarageCars'].fillna(all_data['GarageCars'].mode()[0])

    # category feature with float64 type transform to string
    for col in con_cate:
        all_data[col] = all_data[col].apply(str)

    # Distribution skewness of numerical data(continuous feature)
    skew_feature = all_data[continuous].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skewness': skew_feature})

    # Log conversion of data that does not conform to a normal distribution
    skewness = skewness[abs(skew_feature) > 0.75]
    skew_feature = skewness.index
    lam = 0.15
    for f in skew_feature:
        all_data[f] = boxcox1p(all_data[f], lam)

    # one hot code for category feature
    for col in category:
        feats = pd.get_dummies(all_data[col], prefix=col)
        all_data.drop([col], axis=1, inplace=True)
        all_data = pd.concat([all_data, feats], axis=1)

    for col in con_cate:
        feats = pd.get_dummies(all_data[col], prefix=col)
        all_data.drop([col], axis=1, inplace=True)
        all_data = pd.concat([all_data, feats], axis=1)

    x = all_data[:train_num]
    test = all_data[train_num:]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6,random_state=3)
    lr_y=model_lr=linear_regression(x_train, x_test, y_train, y_test)
    kr_y=model_krg=kernel_ridge(x_train, x_test, y_train, y_test)
    ls_y=model_ls=lasso_regression(x_train, x_test, y_train, y_test)

    plt.plot(np.arange(x_test.shape[0]), y_test, color='k', label='True Value')
    plt.plot(np.arange(x_test.shape[0]), lr_y, color='r', label='Linear')
    plt.plot(np.arange(x_test.shape[0]), kr_y, color='b', label='Kernel Ridge')
    plt.plot(np.arange(x_test.shape[0]), ls_y, color='g', label='Lasso')
    plt.title('regression result comparison')  # title
    plt.legend(loc='upper right')  # picture position
    plt.ylabel('real and predicted value')  # y axis
    plt.show()  # to show the picture
