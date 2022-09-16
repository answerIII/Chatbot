import os
import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout


class NNWrapper:
    def __init__(self):
        self.model = None

        self.model_path = os.path.join(os.curdir, '../models', 'my_model')
        if os.path.exists(os.path.join(self.model_path)):
            self.model = tf.keras.models.load_model('../models/my_model')

    def fit(self, train_X, train_y):

        input_shape = (len(train_X[0]),)
        output_shape = len(train_y[0])
        epochs = 200

        # NN config
        model = Sequential()
        model.add(Dense(128, input_shape=input_shape, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(output_shape, activation="softmax"))
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

        print(model.summary())
        model.fit(x=train_X, y=train_y, epochs=epochs, verbose=1)

        self.model = model
        model.save('../models/my_model')

    def transform(self, data):
        return self.model.predict(data)
