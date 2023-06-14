from asyncio import sleep
from telebot import types
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

from math import floor,ceil,sqrt
import sys
import warnings
import datetime as dt
from pathlib import Path
if not sys.warnoptions:
    warnings.simplefilter("ignore")
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

import telebot

bot = telebot.TeleBot('')
model = 'Linear Regression'
ratio = 0.8
stock = 'AAPL'
def get_dataset(name):
    if os.path.isfile(f'./datasets/stocks/{name}.csv'):
        return pd.read_csv(f'./datasets/stocks/{name}.csv')
    else:
        return None
df = get_dataset(stock)
@bot.message_handler(commands=['start', 'help'])
def start(message):
    mess = f'Привет <b>{message.from_user.first_name}</b>. Краткая информацию по боту:\n' \
           f'Чтобы задать акцию нужно ввести её тикер. Тикер — краткое название в биржевой информации котируемых инструментов (акций, облигаций, индексов). Например Apple=AAPL\n' \
           f'Сущесвтуют 4 модели: Linear Regression, K-Nearest Neighbours, Long Short Term Memory (LSTM)\n' \
           f'Linear Regression - это метод анализа данных, который предсказывает ценность неизвестных данных с помощью другого связанного и известного значения данных\n' \
           f'K-Nearest Neighbours - метрический алгоритм для автоматической классификации объектов или регрессии\n' \
           f'Long Short Term Memory (LSTM) – особая разновидность архитектуры рекуррентных нейронных сетей, способная к обучению долговременным зависимостям\n' \
           f'А ratio с какого момента мы хотим посмотреть'
    bot.send_message(message.chat.id, mess, parse_mode='html')

    commands(message)

@bot.message_handler(commands=['commands'])
def commands(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    setStock = types.KeyboardButton('Задать акцию')
    help = types.KeyboardButton('/help')
    setModel = types.KeyboardButton('Выбрать модель')
    getGraph = types.KeyboardButton('Вывыести график')
    getPredict = types.KeyboardButton('Сделать просчёт модели')
    setRatio = types.KeyboardButton('Задать коэффициент')
    markup.add(help, getGraph, setStock, setModel, getPredict, setRatio)
    bot.send_message(message.chat.id, 'Выберите одну из комманд', reply_markup=markup)
@bot.message_handler()
def get_user_text(message):
    if message.text == 'Выбрать модель':
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
        LinearRegression = types.KeyboardButton('Linear Regression')
        KNearest = types.KeyboardButton('K-Nearest Neighbours')
        LSTM = types.KeyboardButton('LSTM')
        markup.add(LinearRegression, KNearest, LSTM)
        bot.send_message(message.chat.id, 'Выберите нужную модель', reply_markup=markup)
    if message.text == 'Linear Regression' or message.text == 'K-Nearest Neighbours' or message.text == 'LSTM':
        global model
        model = message.text
        bot.send_message(message.chat.id, f'Задана модель {model}', parse_mode='html')
        commands(message)
    if message.text == 'Задать акцию':
        setStock(message)
    if message.text == 'Вывыести график':
        draw_graph(message)
    if message.text == 'Сделать просчёт модели':
        get_prediction(message)
    if message.text == 'Задать коэффициент':
        setRatio(message)

def set_stock(message):
    global stock
    data = get_dataset(message.text)
    if data is not None:
        global df
        df = data
        stock = message.text
        get_dataset_fig(data, message.text)
        photo = open('test.png', 'rb')
        bot.send_photo(message.chat.id, photo)
    else:
        setStock(message)

def setStock(message):
    bot.send_message(message.chat.id, 'Введите тикер:')
    bot.register_next_step_handler(message, set_stock)

def set_ratio(message):
    try:
        if 1 > float(message.text) > 0 :
            global ratio
            ratio = float(message.text)
            bot.send_message(message.chat.id, f'Задан  коэффициент {float(message.text)}', parse_mode='html')
        else:
            setRatio(message)
    except:
        setRatio(message)

def setRatio(message):
    bot.send_message(message.chat.id, 'Введите число от 0 до 1:')
    bot.register_next_step_handler(message, set_ratio)

def draw_graph(message):
    global df, stock
    if df is not None:
        get_dataset_fig(df, stock)
        photo = open('test.png', 'rb')
        bot.send_photo(message.chat.id, photo)
    else:
        get_dataset(stock)


def get_dataset_fig(df, name):
    df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
    df.index = df['Date']
    plt.figure(figsize=(16, 8))
    plt.plot(df['Close'], label='Close Price history', color='g')
    plt.xlabel('Дата', size=20)
    plt.ylabel('Цена акции', size=20)
    plt.title(f'Цена акции {name} за всё время', size=25)
    fname = './test.png'
    plt.savefig(fname)

def get_prediction(message):
    global model, ratio, stock, df
    df = get_dataset(stock)
    df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
    df.index = df['Date']
    Path(f'./{message.from_user.id}').mkdir(parents=True, exist_ok=True)
    plt.cla()
    match model:
        case 'Linear Regression':
            linear_regression_prediction(model, df, ratio, message, stock)
        case 'K-Nearest Neighbours':
            k_nearest_neighbours_predict(model, df, ratio, message, stock)
        case 'LSTM':
           lstm_prediction(model, df, ratio, message, stock)
def linear_regression_prediction(nameModel, df, ratio, message, stock):
    shape = df.shape[0]
    df_new = df[['Close']]
    df_new.head()
    train_set = df_new.iloc[:ceil(shape * ratio)]
    valid_set = df_new.iloc[ceil(shape * ratio):]
    bot.send_message(message.chat.id, 'Прогнозирование цены на акцию с помощью линейной регрессии')
    bot.send_message(message.chat.id, f'Форма обучающего множества {train_set.shape}')
    bot.send_message(message.chat.id, f'Форма валидационного набора {valid_set.shape}')
    train = train_set.reset_index()
    valid = valid_set.reset_index()
    x_train = train['Date'].map(dt.datetime.toordinal)
    y_train = train[['Close']]
    x_valid = valid['Date'].map(dt.datetime.toordinal)
    y_valid = valid[['Close']]
    # implement linear regression
    model = LinearRegression()
    model.fit(np.array(x_train).reshape(-1, 1), y_train)
    preds = model.predict(np.array(x_valid).reshape(-1, 1))
    rms = np.sqrt(np.mean(np.power((np.array(valid_set['Close']) - preds), 2)))
    bot.send_message(message.chat.id, f'Значение RMSE на валидационном множестве: {rms}')
    valid_set['Predictions'] = preds
    plt.plot(train_set['Close'])
    plt.plot(valid_set[['Close', 'Predictions']])
    plt.xlabel('Дата', size=20)
    plt.ylabel('Цена акции', size=20)
    plt.title('Спрогнозированые цены акций с помощью линейной регрессии', size=20)
    plt.legend(['Модель Обучающие данные', 'Акутальные данные', 'Предсказанные данные'])
    fname = f'./{message.from_user.id}/{nameModel}_{message.from_user.id}_{stock}.png'
    plt.savefig(fname)
    sendGraph(fname, message)

def k_nearest_neighbours_predict(nameModel, df, ratio, message, stock):
    shape=df.shape[0]
    df_new=df[['Close']]
    df_new.head()
    train_set=df_new.iloc[:ceil(shape*ratio)]
    valid_set=df_new.iloc[ceil(shape*ratio):]
    bot.send_message(message.chat.id, 'Прогнозирование цены на акцию с помощью метода k-ближайших соседей')
    bot.send_message(message.chat.id, f'Форма обучающего множества {train_set.shape}')
    bot.send_message(message.chat.id, f'Форма валидационного набора {valid_set.shape}')
    train=train_set.reset_index()
    valid=valid_set.reset_index()
    x_train = train['Date'].map(dt.datetime.toordinal)
    y_train = train[['Close']]
    x_valid = valid['Date'].map(dt.datetime.toordinal)
    y_valid = valid[['Close']]
    x_train_scaled = scaler.fit_transform(np.array(x_train).reshape(-1, 1))
    x_train = pd.DataFrame(x_train_scaled)
    x_valid_scaled = scaler.fit_transform(np.array(x_valid).reshape(-1, 1))
    x_valid = pd.DataFrame(x_valid_scaled)
    params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
    knn = neighbors.KNeighborsRegressor()
    model = GridSearchCV(knn, params, cv=5)
    model.fit(x_train,y_train)
    preds = model.predict(x_valid)
    rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
    bot.send_message(message.chat.id, f'Значение RMSE на валидационном множестве: {rms}')
    valid_set['Predictions'] = preds
    plt.plot(train_set['Close'])
    plt.plot(valid_set[['Close', 'Predictions']])
    plt.xlabel('Дата', size=20)
    plt.ylabel('Цена акции', size=20)
    plt.title('Спрогнозированые цены акций с помощью метода K-ближайших соседей', size=20)
    plt.legend(['Модель Обучающие данные', 'Акутальные данные', 'Предсказанные данные'])
    fname = f'./{message.from_user.id}/{nameModel}_{message.from_user.id}_{stock}.png'
    plt.savefig(fname)
    sendGraph(fname, message)

def lstm_prediction(nameModel, df, ratio, message, stock):
    shape = df.shape[0]
    df_new = df[['Close']]
    df_new.head()
    dataset = df_new.values
    train = df_new[:ceil(shape * ratio)]
    valid = df_new[ceil(shape * ratio):]
    bot.send_message(message.chat.id, 'Прогнозирование цены на акцию с помощью метода LSTM')
    bot.send_message(message.chat.id, f'Форма обучающего множества {train.shape}')
    bot.send_message(message.chat.id, f'Форма валидационного набора {valid.shape}')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    x_train, y_train = [], []
    for i in range(40, len(train)):
        x_train.append(scaled_data[i - 40:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
    inputs = df_new[len(df_new) - len(valid) - 40:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    X_test = []
    for i in range(40, inputs.shape[0]):
        X_test.append(inputs[i - 40:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)
    rms = np.sqrt(np.mean(np.power((valid - closing_price), 2)))
    bot.send_message(message.chat.id, f'Значение RMSE на валидационном множестве: {rms}')
    valid['Predictions'] = closing_price
    plt.plot(train['Close'])
    plt.plot(valid['Close'])
    plt.xlabel('Дата', size=20)
    plt.ylabel('Цена акции', size=20)
    plt.title('График актуальной цены', size=20)
    plt.legend(['Модель Обучающие данные', 'Акутальные данные'])
    fname = f'./{message.from_user.id}/{nameModel}_{message.from_user.id}_{stock}_1.png'
    plt.savefig(fname)
    sendGraph(fname, message)
    plt.cla()
    plt.plot(train['Close'])
    plt.plot(valid['Predictions'])
    plt.xlabel('Дата', size=20)
    plt.ylabel('Цена акции', size=20)
    plt.title('Спрогнозированные цены на акции с помощью долговременной краткосрочной памяти(LSTM)', size=20)
    plt.legend(['Модель Обучающие данные', 'Предсказанные данные'])
    fname = f'./{message.from_user.id}/{nameModel}_{message.from_user.id}_{stock}_2.png'
    plt.savefig(fname)
    sendGraph(fname, message)
    plt.cla()
    plt.plot(train['Close'])
    plt.plot(valid['Close'])
    plt.plot(valid['Predictions'])
    plt.xlabel('Дата', size=20)
    plt.ylabel('Цена акции', size=20)
    plt.title('Спрогнозированные цены на акции с помощью долговременной краткосрочной памяти(LSTM)', size=20)
    plt.legend(['Модель Обучающие данные', 'Акутальные данные', 'Предсказанные данные'])
    fname = f'./{message.from_user.id}/{nameModel}_{message.from_user.id}_{stock}.png'
    plt.savefig(fname)
    sendGraph(fname, message)

def sendGraph(path, message):
    if os.path.isfile(path):
        photo = open(path, 'rb')
        bot.send_photo(message.chat.id, photo)
bot.polling(none_stop=True)