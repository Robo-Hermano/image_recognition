#in blocks again to show how I ran them
#prepatory imports and data adjusting
from tensorflow import keras
from keras.datasets import cifar10
import numpy as np
seedy = 21
(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()
xtrain = xtrain.astype('float32')
xtest = xtest.astype('float32')
xtrain = xtrain / 255.0
xtest = xtest / 255.0
ytrain = keras.utils.to_categorical(ytrain)
ytest = keras.utils.to_categorical(ytest)
class_num = ytest.shape[1]



#now build model
my_model = keras.models.Sequential([
    keras.layers.Conv2D(32, 3, input_shape = xtrain.shape[1:], padding = 'same', activation = 'relu'),
    keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same'),
    keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),

    keras.layers.Flatten(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation = 'relu'),
    keras.layers.Dropout(0.3),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(class_num, activation = 'softmax')
])

my_model.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)

print(my_model.summary())



np.random.seed(seedy)
my_model.fit(xtrain, ytrain, epochs = 10, batch_size = 64, validation_data = (xtest, ytest))



score = my_model.evaluate(xtest, ytest, verbose = 0)
print(score[0])
