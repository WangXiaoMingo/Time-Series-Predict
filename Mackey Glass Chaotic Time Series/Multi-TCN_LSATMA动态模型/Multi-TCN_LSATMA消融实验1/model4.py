import pandas as pd
from pylab import *
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,LSTM,Input,Concatenate,Activation,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,LearningRateScheduler
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from attention1 import Attention1
import tensorflow as tf
from tcn import TCN, tcn_full_summary
from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,mean_absolute_percentage_error,explained_variance_score
import warnings
warnings.filterwarnings("ignore")

font1 = {'family': 'Times New Roman','weight': 'normal','size': 13,}
font2 = {'family': 'STSong','weight': 'normal','size': 13,}
fontsize1=13

# 设置字体，以作图显示中文
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus']=False
# 设置显示属性
pd.set_option('display.max_columns',500)
pd.set_option('display.max_rows',100)
np.set_printoptions(suppress=True)
pd.set_option('precision',5)
np.set_printoptions(precision=5)

# 以下两句速度慢
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "/device:CPU:0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "/device:GPU:0"


def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 200 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.5)
        print("lr changed to {}".format(lr * 0.5))
    return K.get_value(model.optimizer.lr)

def figure_plot(predict, true_value, figure_property,key_label=None):
    # 折线图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(predict, '-', label='预测值')
    ax.plot(true_value, '-', label='真实值')
    # x_ticks = ax.set_xticks([i for i in range(len(key_label))])
    # x_labels = ax.set_xticklabels(key_label,rotation=45,fontdict=font1)
    ax.set_title(figure_property['title'], fontdict=font2)
    ax.set_xlabel(figure_property['X_label'], fontdict=font2)
    ax.set_ylabel(figure_property['Y_label'], fontdict=font2)
    plt.tick_params(labelsize=12)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # #     y_ticks=ax.set_yticks([])
    # #     y_labels=ax.set_yticklabels([-20+i for i in range(20)],rotation=0,fontsize=14)
    plt.legend(prop=font2)
    plt.tight_layout()
    # plt.savefig('../fig/{}.jpg'.format(figure_property['title']), dpi=500, bbox_inches='tight')  # 保存图片
    plt.show()

def figure_plot_1(predict, true_value, figure_property,key_label=None):
    # 折线图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(true_value,predict, '-*')
    # x_ticks = ax.set_xticks([i for i in range(len(key_label))])
    # x_labels = ax.set_xticklabels(key_label,rotation=45,fontdict=font1)
    ax.set_title(figure_property['title'], fontdict=font2)
    ax.set_xlabel(figure_property['X_label'], fontdict=font2)
    ax.set_ylabel(figure_property['Y_label'], fontdict=font2)
    plt.tick_params(labelsize=12)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.legend()
    plt.tight_layout()
    # plt.savefig('../fig/{}.jpg'.format(figure_property['title']), dpi=500, bbox_inches='tight')  # 保存图片
    plt.show()

def Calculate_Regression_metrics(true_value, predict, label='训练集'):
    mse = mean_squared_error(true_value, predict)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_value, predict)
    r2 = r2_score(true_value, predict)
    ex_var = explained_variance_score(true_value, predict)
    mape = mean_absolute_percentage_error(true_value, predict)
    train_result = pd.DataFrame([mse, rmse, mae, r2,ex_var,mape], columns=[label],
                                index=['mse', 'rmse', 'mae', 'r2','ex_var','mape']).T
    return train_result

# TODO: 定义模型，如图片所示
def Model4():
    '''LSTM和GRU、Attention进行融合后进行输出'''
    input1 = Input(shape=(window, int(x_dim/window)))
    # 网络1
    x1 = LSTM(128, return_sequences=True)(input1)
    x1 = LSTM(64, return_sequences=True)(x1)
    x1 = Attention1(units=64)(x1)
    # 网络2
    x2 = LSTM(128, return_sequences=True)(input1)
    x2 = LSTM(64, return_sequences=True)(x2)
    x2 = Attention1(units=64)(x2)
    # 特征融合
    x = Concatenate(axis=1)([x1, x2])
    # 回归层
    x = Dense(units=64)(x)
    out = Dense(units=1, activation='linear')(x)
    model = Model(inputs=input1, outputs=out)
    return model

# TODO: 1.加载数据
'''============================ 1. 加载数据 ============================'''

'''=============1. 读取数据======================='''
print("---开始运行--加载数据------------------------")
data = pd.read_excel('./data/Mackey_Glass_Chaotic_Time_Series.xlsx',header=None)
print(data)
train_data = data.iloc[:500,:]
test_data = data.iloc[500:,:]
time_name = test_data.index.values
print("---加载数据完成------------------------")

X_train = train_data.iloc[:,:-1].values
Y_train = train_data.iloc[:,-1].values.reshape(-1,1)

X_test = test_data.iloc[:,:-1].values
Y_test = test_data.iloc[:,-1].values.reshape(-1,1)

import gc
del train_data,test_data
gc.collect()

print('数据形状：X_train:{}，X_test:{},Y_train:{},Y_test:{}'.format(X_train.shape, X_test.shape,
                                                                          Y_train.shape, Y_test.shape))
'''=============2. 数据预处理======================='''
scaler_x = MinMaxScaler(feature_range=(-1, 1))
scaler_x = scaler_x.fit(X_train)
X_train = scaler_x.transform(X_train)
X_test = scaler_x.transform(X_test)

scaler_y = MinMaxScaler(feature_range=(-1, 1))
scaler_y = scaler_y.fit(Y_train)
y_train = scaler_y.transform(Y_train)
y_test = scaler_y.transform(Y_test)

window = 1
x_dim = X_train.shape[1]
y_dim = Y_train.shape[1]
X_train = X_train.reshape(-1,window,int(x_dim/window))
X_test = X_test.reshape(-1,window,int(x_dim/window))

'''=============3. 训练模型======================='''
algorithm = 'model4'
print(f'======================{algorithm}=========================')
# 多次训练取均值，以评价模型性能，均值和标准差
for i in range(1):
    a = time.time()
    model = Model4()
    model.summary()#展示模型结构
    # 保存每次训练过程中的最佳的训练模型
    # 有一次提升, 则覆盖一次.
    checkpoint = ModelCheckpoint(f'./temp/{algorithm}.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                 mode='min')
    # 当评价指标不在提升时，减少学习率
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=50, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    reduce_lr = LearningRateScheduler(scheduler)
    callbacks_list = [checkpoint,reduce_lr]
    optimer = Adam(lr=0.01)  # 0.001->0.011; 0.005-> 0.0047; 0.01->0.00433 0.05->0.008; 0.1->0.06
    model.compile(loss='mse', optimizer=optimer, metrics=['mse'])

    epoch = 500
    history = model.fit(X_train, y_train, epochs=epoch, batch_size=128, shuffle=False,
                        validation_data=(X_test, y_test), callbacks=callbacks_list)  # 训练模型epoch次

    b = time.time() - a
    print(f'第{i + 1}次程序运行时间{b}')

    # 加载最优模型
    del model
    model = load_model(f'./temp/{algorithm}.h5' ,custom_objects={'Attention1':Attention1,'TCN': TCN})

    '''=============4. 测试模型性能======================='''
    train_predict = scaler_y.inverse_transform(model.predict(X_train))
    test_predict = scaler_y.inverse_transform(model.predict(X_test))

    #迭代图像
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epoch)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Test Loss')
    plt.legend(loc='upper right')
    plt.title(f'{algorithm}Train and Val Loss')
    plt.show()

    '''================计算模型计算训练集结果=============='''
    train_result = Calculate_Regression_metrics(Y_train, train_predict, label='训练集')
    # title = '{}算法训练集结果对比'.format(algorithm)
    # figure_property = {'title': title, 'X_label': 'Prediction set samples', 'Y_label': 'Prediction Value'}
    # figure_plot(train_predict, Y_train, figure_property)

    '''================计算模型计算测试集结果=============='''
    test_result = Calculate_Regression_metrics(Y_test, test_predict, label='测试集')
    # title = '{}算法测试集结果对比'.format(algorithm)
    # figure_property = {'title': title, 'X_label': 'Prediction set samples', 'Y_label': 'Prediction Value'}
    # figure_plot(test_predict, Y_test, figure_property)

    '''===================保存计算结果================'''
    result = pd.concat([train_result, test_result], axis=0)
    print('\n {}算法第{}次计算结果'.format(algorithm,i+1))
    print(result)

    # 保存结果
    test_result['time'] = b
    test_result.index = [f'第{i+1}次']

    # 保存模型
    if i == 0:
        all_result = test_result
        pred_y = pd.DataFrame([Y_test.flatten(), test_predict.flatten()], index=['真实值', '预测值1'], columns=time_name).T
        error_mse = test_result['mse'].values
        best_model = model
    else:
        all_result = pd.concat([all_result, test_result], axis=0)
        pred_y[f'预测值{i + 1}'] = test_predict.flatten()

        if test_result['mse'].values < error_mse:
            error_mse = test_result['mse'].values
            best_model = model
    del model,result,history,train_predict,test_predict,train_result,test_result
    gc.collect()
    print(all_result)
all_result.loc['平均值'] = np.mean(all_result, axis=0)
print(all_result)
best_model.save(f'./model/{algorithm}.h5')
all_result.to_excel('./result/{}算法计算结果均值.xlsx'.format(algorithm))
# pred_y.to_excel('./result/{}算法计算结果多次运行预测值.xlsx'.format(algorithm))

