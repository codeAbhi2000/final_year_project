import pandas as pd
import numpy as np
from scipy.stats import  pearsonr  
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

def pricePrridict(inData):


    dataset = pd.read_excel('D:/Python programs/crops_prices.xlsx')

# print(dataset)

    features = dataset.iloc[:,:-3]

    features = features.drop('date_arrival',axis='columns')
    features = features.drop('state_name',axis='columns')

# trainX = np.asarray(features[:9500])

    # print(features)


    targets = dataset.iloc[:,-2]

# trainY = np.asarray(targets[:9500])

    # print(targets)

    model = linear_model.LinearRegression()

    # e_stateName = LabelEncoder()
    # features.state_name = e_stateName.fit_transform(features.state_name)
    e_distName = LabelEncoder()
    features.district_name = e_distName.fit_transform(features.district_name)
    e_marketName = LabelEncoder()
    features.market_center_name = e_marketName.fit_transform(features.market_center_name)
    e_variety = LabelEncoder()
    features.Variety = e_variety.fit_transform(features.Variety)
    e_grpName = LabelEncoder()
    features.group_name = e_grpName.fit_transform(features.group_name)
    # e_dateArrival = LabelEncoder()
    # features.date_arrival = e_dateArrival.fit_transform(features.date_arrival)


# print(features)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features,targets, test_size = 0.20)

# plt.scatter(trainX,trainY)
# plt.plot(X_train,y_train)

# plt.show()

# print(X_train.corr())

# plt.figure(figsize=(12,10))

# plt.scatter(features['date_arrival'],targets)
# plt.show()

    cor ,_= pearsonr(features['district_name'],targets)
    print(cor)

# print(X_test['district_name'])

    # model.fit(X_train,y_train)

    # result2 = model.predict(inData)

# print(result)

# print("Mean squared error is: " ,mean_squared_error(y_test,result))

    randModel = RandomForestRegressor(n_estimators = 1000, random_state = 42)

    randModel.fit(features,targets)

    # print(X_train)

    result = randModel.predict(inData)

    return result 
    '''
    error = abs(result-y_test)
    print('Mean Absolute Error:', round(np.mean(error), 2), 'degrees.')
    mape = 100 * (error / y_test)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
    print("Mean squared error is: " ,mean_squared_error(y_test,result2))
    '''



print("Enter the Data for Prediction")

li = ['Bnagalore','Bangalore','Beans','Vegetables','04','23']

print(li)

di = {
    'district_name':'',
    'market_center_name':'',
    'Variety':'',
    'group_name':'',
    'Month':'',
    'Year':'',
}

for i in range(len(li)):
    di[list(di.keys())[i]] = li[i]

df = pd.DataFrame(di,index=[0])



e_distName = LabelEncoder()
df.district_name = e_distName.fit_transform(df.district_name)
e_marketName = LabelEncoder()
df.market_center_name = e_marketName.fit_transform(df.market_center_name)
e_variety = LabelEncoder()
df.Variety = e_variety.fit_transform(df.Variety)
e_grpName = LabelEncoder()
df.group_name = e_grpName.fit_transform(df.group_name)



result = pricePrridict(df)[0]

print(round(result,2))
