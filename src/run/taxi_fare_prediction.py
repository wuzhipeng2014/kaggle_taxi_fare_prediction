#coding:utf-8

import  pandas as pd
import  numpy as np
import  os


train_df=pd.read_csv('../data/train.csv',nrows=300000,parse_dates=["pickup_datetime"])

# print (train_df.dtypes)
#
# print (train_df.head())


print(train_df.describe())

view=train_df[train_df==0].count()
print(view)


print[train_df.loc[314]]



train_df=train_df[train_df.fare_amount>0]
train_df=train_df[train_df.pickup_longitude<180]
train_df=train_df[train_df.pickup_longitude>-180]
train_df=train_df[train_df.pickup_latitude>-90]
train_df=train_df[train_df.pickup_latitude<90]
train_df=train_df[train_df.dropoff_longitude<180]
train_df=train_df[train_df.dropoff_longitude>-180]
train_df=train_df[train_df.dropoff_latitude>-90]
train_df=train_df[train_df.dropoff_latitude<90]


print(train_df.shape)

print (train_df['fare_amount'].std(axis=0))

print(train_df[train_df.fare_amount>100].count())

import matplotlib.pyplot as plt

train_df[train_df.fare_amount<100].fare_amount.hist(bins=100, figsize=(14,3))
plt.xlabel('fare $USD')
plt.title('Histogram')

plt.show()

## drop na value

print('train_df.isnull().sum(): %s' %train_df.isnull().sum())


train_df= train_df.dropna(how='any',axis='rows')


# This function is based on https://stackoverflow.com/questions/27928/
# calculate-distance-between-two-latitude-longitude-points-haversine-formula
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 12742 * np.arcsin(np.sqrt(a)) # 2*R*asin...


train_df['distance_km']=distance(train_df['pickup_latitude'],train_df['pickup_longitude'],train_df['dropoff_latitude'],train_df['dropoff_longitude'])

print(train_df['distance_km'].describe())

print('## 出发 终点 距离小于1km的地点')

## todo 可以查看此部分行程的开始 终止 点 有几种
print(train_df[train_df['distance_km']<1].head())




train_df=train_df[train_df['distance_km']>1]



print("出行距离大于100km的行程数目为:%s" %train_df[train_df['distance_km']>100].distance_km.count())

train_df=train_df[train_df['distance_km']<100]


train_df[train_df['distance_km']<100].distance_km.hist(bins=100, figsize=(14,3))

plt.xlabel('distance_km')
plt.ylabel('trip count')

plt.show()

sum_result=train_df[['fare_amount','distance_km']].apply(sum,axis=0)

print(sum_result)
print ('数据集上每千米的平均价格为: %s' %(sum_result.fare_amount/sum_result.distance_km))


print("Average $USD/KM : {:0.2f}".format(train_df.fare_amount.sum()/train_df.distance_km.sum()))

print( train_df.groupby('passenger_count')['distance_km','fare_amount'].mean() )




# scatter plot distance - fare
fig, axs = plt.subplots(1, 2, figsize=(16,6))
axs[0].scatter(train_df.distance_km, train_df.fare_amount, alpha=0.2)
axs[0].set_xlabel('distance km')
axs[0].set_ylabel('fare $USD')
axs[0].set_title('All data')

# zoom in on part of data
idx = (train_df.distance_km < 21) & (train_df.fare_amount < 100)
axs[1].scatter(train_df[idx].distance_km, train_df[idx].fare_amount, alpha=0.2)
axs[1].set_xlabel('distance km')
axs[1].set_ylabel('fare $USD')
axs[1].set_title('Zoom in on distance < 20km, fare < $100');

plt.show()


## remove pickup and dropoff place within 100m

train_df=train_df[train_df['distance_km']>0.1]



## add time detail info to train_df

train_df['hour']=train_df['pickup_datetime'].apply(lambda x:x.hour)
train_df['year']=train_df['pickup_datetime'].apply(lambda x:x.year)
train_df['fare_per_km']=train_df['fare_amount']/train_df['distance_km']

print("train_df.groupby('hour')['fare_per_km'].describe(): ")
print(train_df.groupby('hour')['fare_per_km'].describe())


print("train_df['fare_per_km'].describe(): \n %s" %train_df['fare_per_km'].describe())


train_df.pivot_table('fare_per_km', index='hour', columns='year').plot(figsize=(14,6))


plt.ylabel('Fare $USD / KM')
plt.show()

print('## 经纬度保留两位小数,精确到千米')

train_df['pickup_longitude']=train_df['pickup_longitude'].apply(lambda x:'%.2f' %x).astype(str)
train_df['pickup_latitude']=train_df['pickup_latitude'].apply(lambda x:'%.2f' %x).astype(str)
train_df['dropoff_longitude']=train_df['dropoff_longitude'].apply(lambda x:'%.2f' %x).astype(str)
train_df['dropoff_latitude']=train_df['dropoff_latitude'].apply(lambda x:'%.2f' %x).astype(str)
print(train_df.dtypes)


# -----------------处理测试集合上的数据--------------------

submission_df=pd.read_csv('../data/test.csv',parse_dates=["pickup_datetime"])

submission_df=submission_df[submission_df.pickup_longitude<180]
submission_df=submission_df[submission_df.pickup_longitude>-180]
submission_df=submission_df[submission_df.pickup_latitude>-90]
submission_df=submission_df[submission_df.pickup_latitude<90]
submission_df=submission_df[submission_df.dropoff_longitude<180]
submission_df=submission_df[submission_df.dropoff_longitude>-180]
submission_df=submission_df[submission_df.dropoff_latitude>-90]
submission_df=submission_df[submission_df.dropoff_latitude<90]

submission_df= submission_df.dropna(how='any',axis='rows')



submission_df['distance_km']=distance(submission_df['pickup_latitude'],submission_df['pickup_longitude'],submission_df['dropoff_latitude'],submission_df['dropoff_longitude'])


submission_df['hour']=submission_df['pickup_datetime'].apply(lambda x:x.hour)
submission_df['year']=submission_df['pickup_datetime'].apply(lambda x:x.year)
# submission_df['fare_per_km']=submission_df['fare_amount']/submission_df['distance_km']

submission_df['pickup_longitude']=submission_df['pickup_longitude'].apply(lambda x:'%.2f' %x).astype(str)
submission_df['pickup_latitude']=submission_df['pickup_latitude'].apply(lambda x:'%.2f' %x).astype(str)
submission_df['dropoff_longitude']=submission_df['dropoff_longitude'].apply(lambda x:'%.2f' %x).astype(str)
submission_df['dropoff_latitude']=submission_df['dropoff_latitude'].apply(lambda x:'%.2f' %x).astype(str)
print(submission_df.dtypes)


submission_dummy_train_feature=pd.get_dummies(submission_df[['pickup_longitude', 'pickup_latitude',  'dropoff_longitude' ,'dropoff_latitude' ,'passenger_count' ,'distance_km','hour' ,'year']])


train_dummpy_feature_tmp=train_df[[ 'pickup_longitude', 'pickup_latitude',  'dropoff_longitude' ,'dropoff_latitude']]
submission_dummpy_feature_tmp=submission_df[[ 'pickup_longitude', 'pickup_latitude',  'dropoff_longitude' ,'dropoff_latitude']]

concat_dummpy_feature_df=pd.concat([train_dummpy_feature_tmp,submission_dummpy_feature_tmp])



concat_dummpy_feature_df=pd.get_dummies(pd.DataFrame(concat_dummpy_feature_df))

print('concat_dummpy_feature_df.shape=')
print(pd.DataFrame(concat_dummpy_feature_df).shape)

train_dummy_feature=concat_dummpy_feature_df[:train_dummpy_feature_tmp.shape[0]]

submission_dummpy_feature=concat_dummpy_feature_df[train_dummpy_feature_tmp.shape[0]:]
print(train_dummy_feature.shape)
print(submission_dummpy_feature.shape)

train_df_int_feature=train_df[['passenger_count','distance_km','hour' ,'year']]

final_train_dummpy_feature=pd.DataFrame(pd.concat([train_df_int_feature,train_dummy_feature],axis=1))

submission_df_int_feature=submission_df[['passenger_count','distance_km','hour' ,'year']]

final_submission_dummpy_feature=pd.DataFrame(pd.concat([submission_df_int_feature,submission_dummpy_feature],axis=1))

print('final  dummpy feature shape=')
print(pd.DataFrame(final_train_dummpy_feature).shape)
print (pd.DataFrame(final_submission_dummpy_feature).shape)






# ------------------------------------------------------


dummy_train_feature=pd.get_dummies(train_df[['fare_amount', 'pickup_longitude', 'pickup_latitude',  'dropoff_longitude' ,'dropoff_latitude' ,'passenger_count' ,'distance_km','hour' ,'year', 'fare_per_km']])





# print('dummy_train_feature.shape: %s' %dummy_train_feature.shape)
print(dummy_train_feature.shape)




print('##起始点精度统计')
print(train_df.groupby('pickup_longitude')['pickup_longitude'].count())

print(train_df['pickup_longitude'].unique().__len__())




print('最终训练集中的特征名称为: %s' %train_df.columns.values)

# ['key' 'fare_amount' 'pickup_datetime' 'pickup_longitude' 'pickup_latitude'
#  'dropoff_longitude' 'dropoff_latitude' 'passenger_count' 'distance_km'
 # 'hour' 'year' 'fare_per_km']

print(train_df.head())

print('最终训练集中的特征名称为: %s' %dummy_train_feature.columns.values)

print(dummy_train_feature.head())

feature_list=dummy_train_feature.columns.values.tolist()


# define some handy analysis support function
from sklearn.metrics import mean_squared_error, explained_variance_score

def plot_prediction_analysis(y, y_pred, figsize=(10,4), title=''):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].scatter(y, y_pred)
    mn = min(np.min(y), np.min(y_pred))
    mx = max(np.max(y), np.max(y_pred))
    axs[0].plot([mn, mx], [mn, mx], c='red')
    axs[0].set_xlabel('$y$')
    axs[0].set_ylabel('$\hat{y}$')
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    evs = explained_variance_score(y, y_pred)
    axs[0].set_title('rmse = {:.2f}, evs = {:.2f}'.format(rmse, evs))

    axs[1].hist(y-y_pred, bins=50)
    avg = np.mean(y-y_pred)
    std = np.std(y-y_pred)
    axs[1].set_xlabel('$y - \hat{y}$')
    axs[1].set_title('Histrogram prediction error, $\mu$ = {:.2f}, $\sigma$ = {:.2f}'.format(avg, std))

    if title!='':
        fig.suptitle(title)




## 开始训练模型

features=['fare_amount', 'pickup_longitude', 'pickup_latitude',  'dropoff_longitude' ,'dropoff_latitude' ,'passenger_count' ,'distance_km','hour' ,'year', 'fare_per_km']




from sklearn.model_selection import train_test_split

feature_list.remove('fare_amount')
feature_list.remove('fare_per_km')
print(feature_list)

# X=dummy_train_feature[feature_list].values
X=final_train_dummpy_feature.values

y=dummy_train_feature['fare_amount'].values




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

print('## X_train.shape=%s, X_test.shape=%s'%(X_train.shape,X_test.shape))




from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

model_lin = Pipeline((
    ("standard_scaler", StandardScaler()),
    ("lin_reg", LinearRegression()),
))
model_lin.fit(X_train, y_train)

y_train_pred = model_lin.predict(X_train)
plot_prediction_analysis(y_train, y_train_pred, title='Linear Model - Trainingset')

plt.show()


y_test_pred = model_lin.predict(X_test)
plot_prediction_analysis(y_test, y_test_pred, title='Linear Model - Testset')

test_set_pedict_result_tmp=pd.concat([pd.DataFrame(y_test),pd.DataFrame(y_test_pred)],axis=1)

pd.DataFrame(test_set_pedict_result_tmp).to_csv('../data/test_set_predict_result.csv',index=False)


plt.show()


## 生成提交结果



# train_df=pd.read_csv('../data/test.csv',parse_dates=["pickup_datetime"])
#
# train_df=train_df[train_df.pickup_longitude<180]
# train_df=train_df[train_df.pickup_longitude>-180]
# train_df=train_df[train_df.pickup_latitude>-90]
# train_df=train_df[train_df.pickup_latitude<90]
# train_df=train_df[train_df.dropoff_longitude<180]
# train_df=train_df[train_df.dropoff_longitude>-180]
# train_df=train_df[train_df.dropoff_latitude>-90]
# train_df=train_df[train_df.dropoff_latitude<90]
#
# train_df= train_df.dropna(how='any',axis='rows')
#
#
#
# train_df['distance_km']=distance(train_df['pickup_latitude'],train_df['pickup_longitude'],train_df['dropoff_latitude'],train_df['dropoff_longitude'])
#
#
# train_df['hour']=train_df['pickup_datetime'].apply(lambda x:x.hour)
# train_df['year']=train_df['pickup_datetime'].apply(lambda x:x.year)
# # train_df['fare_per_km']=train_df['fare_amount']/train_df['distance_km']
#
# train_df['pickup_longitude']=train_df['pickup_longitude'].apply(lambda x:'%.2f' %x).astype(str)
# train_df['pickup_latitude']=train_df['pickup_latitude'].apply(lambda x:'%.2f' %x).astype(str)
# train_df['dropoff_longitude']=train_df['dropoff_longitude'].apply(lambda x:'%.2f' %x).astype(str)
# train_df['dropoff_latitude']=train_df['dropoff_latitude'].apply(lambda x:'%.2f' %x).astype(str)
# print(train_df.dtypes)
#
#
# dummy_train_feature=pd.get_dummies(train_df[['pickup_longitude', 'pickup_latitude',  'dropoff_longitude' ,'dropoff_latitude' ,'passenger_count' ,'distance_km','hour' ,'year']])
#
#
# dummy_train_feature=dummy_train_feature[feature_list]
#
# X=dummy_train_feature[feature_list].values
#
#
#
X_submission=final_submission_dummpy_feature.values
y_pred_final = model_lin.predict(X_submission)

y_pred_final[y_pred_final>100]=100
y_pred_final[y_pred_final<0]=0





submission = pd.DataFrame(
    {'key': submission_df.key, 'fare_amount': y_pred_final},
    columns = ['key', 'fare_amount'])
submission.to_csv('../data/submission.csv', index = False)










