import numpy as np
from tensorflow.keras.utils import Sequence

seq = 2
N = 1000

class DataGenerator(Sequence):
    def __init__(self, seq, N, batch_size, to_fit=True):
        self.seq = seq
        self.N = N
        self.to_fit = to_fit
        self.batch_size = batch_size

    def __len__(self):
        return int(self.N / self.batch_size)

    def __getitem__(self, index):
        x = np.array([np.random.normal(0, 1, self.seq) for i in range(self.batch_size)])
        if self.to_fit:
            y = x[:, 0] + x[:, -1]
            return x.reshape(list(x.shape) + [1]), y
        return x

trainGen = DataGenerator(seq, N, 256)
validGen = DataGenerator(seq, int(N*.3), int(256*.3))

from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

X = Input(shape=(seq,1))
CNN_1 = Conv1D(filters=3, kernel_size=seq, activation='softplus', padding='valid')(X)
FLATTEN_1 = Flatten()(CNN_1)
DENSE_1 = Dense(2, activation='softplus')(FLATTEN_1)
Y_HAT = Dense(1)(DENSE_1)
model = Model(inputs=X, outputs=Y_HAT)

print(model.compile(loss='mse'))
print(model.summary())

MODEL_NAME = 'cnn.hdf5'
checkpoint = ModelCheckpoint(MODEL_NAME, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(trainGen, validation_data=validGen, epochs=1000, callbacks=callbacks_list, verbose=0)
model.load_weights(MODEL_NAME)

print(model.evaluate(validGen))

#### Eval #####
from utils import *

print("#"*10, "Eval", "#"*10)
testX, testY = validGen.__getitem__(0)
y_pred = model.predict(testX)

change(testX, 50, y_pred, model, -1)
change(testX, 50, y_pred, model, 0)
blinding(testX, 50, y_pred, model, 0, -1)
blinding(testX, 50, y_pred, model, -1, 0)


