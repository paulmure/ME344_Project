import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import Sequential
import numpy as np
from matplotlib import pyplot as plt


def get_data():
    filename = 'data/processed/cases_and_deaths.csv'
    data = np.genfromtxt(filename, delimiter=',', dtype=int)

    np.random.shuffle(data)

    return data


def get_sliced_data(data):
    result = {}
    
    result['train_x'] = data[:2518690, :20].reshape((-1, 10, 2))
    result['test_x'] = data[2518690:, :20].reshape((-1, 10, 2))
    
    result['train_y'] = data[:2518690, 20]
    result['test_y'] = data[2518690:, 20]

    return result


def make_model():
    model = Sequential()
    model.add(Input(shape=(10, 2)))
    model.add(LSTM(10, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='relu'))
    
    opt = tf.keras.optimizers.Adam()
    
    model.compile(optimizer=opt,
                  loss=MeanSquaredError(),
                  metrics=['mean_squared_error'])
    
    return model


def cp_callback():
    checkpoint_path = 'model/model.ckpt'
    return tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                              save_weights_only=True,
                                              verbose=1)
                                              

def fit_model(model, datas, epochs, callbacks=None, checkpoint_path='model/model.ckpt'):
    inp = datas['train_x']
    out = datas['train_y']

    history = model.fit(inp, out, epochs=epochs, callbacks=callbacks)

    return history


def make_plot(history):
    plt.plot(history.history['mean_squared_error'])
    plt.savefig('history.png')


def test_model(model, datas):
    inp = datas['test_x']
    out = datas['test_y']
    test_loss, test_acc = model.evaluate(inp, out, verbose=2)
    return test_loss, test_acc


def main():
    datas = get_sliced_data(get_data())
    model = make_model()
    
    history = fit_model(model, datas, 3, callbacks=[cp_callback()])
    
    make_plot(history)
    test_loss, test_acc = test_model(model, datas)

    print('test_loss = {}'.format(test_loss))
    print('test_acc = {}'.format(test_acc))


if __name__ == '__main__':
    main()
