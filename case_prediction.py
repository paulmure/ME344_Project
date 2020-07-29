from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

data = np.genfromtxt('processed_cases.csv', delimiter=',')
np.random.shuffle(data)
train_data = data[:322181, :4]
train_labels = data[:322181, 5]
test_data = data[322181:, :4]
test_labels = data[322181:, 5]

model = keras.Sequential([
    keras.layers.Input(shape=(4,)),
    keras.layers.Dense(4, activation='tanh'),
    keras.layers.Dense(8, activation='tanh'),
    keras.layers.Dense(8, activation='tanh'),
    keras.layers.Dense(4, activation='tanh'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=keras.losses.MeanSquaredError(),
              metrics=['mean_squared_error'])

history = model.fit(train_data, train_labels, epochs=1000)
plt.plot(history.history['mean_squared_error'])
plt.savefig('graph.png')

test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)

model.save()