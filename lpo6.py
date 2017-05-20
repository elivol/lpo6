import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from re import sub
import sys

# Ввод данных
print("Введите количество символов для генерации текста:")
while True:
    try:
        chars_for_gen = int(input())
    except ValueError:
        print("Неверный ввод. Введите число!")
    else:
        break

# Загрузка текста и перевод его символовв нижний регистр
filename = 'sherlock.txt'
raw_text = open(filename).read().lower()

# Удаление пунктуации
raw_text = sub('[ .,!?:;\"\'`~@#$%^&*()\\\[\]/]+', ' ', raw_text)

# Выбор уникальных символов и ассоциация их с числами
chars = sorted(list(set(raw_text)))
char_to_integers = dict((c, i) for i, c in enumerate(chars))

# Подсчет кол-ва символов и уникальных символов
n_chars = len(raw_text)
n_vocab = len(chars)
print("Всего символов: ", n_chars)
print("Всего уникальных символов: ", n_vocab)

# Подготовка данных для обучения
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    #
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_integers[char] for char in seq_in])
    dataY.append(char_to_integers[seq_out])
n_patterns = len(dataX)
print("Выделено шаблонов: ", n_patterns)

# X к виду [samples, time steps, features]
X = numpy.array(dataX)
print(X.shape)
X = X.reshape((n_patterns, seq_length, 1))
print(X.shape)
# нормализация X
X = X / float(n_vocab)
# кодирование выходной переменной
y = np_utils.to_categorical(dataY)

# Определение модели LSTM
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

"""model.compile(loss='categorical_crossentropy', optimizer='adam')

# Определение чек-пойнтов
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Обучение сети
model.fit(X, y, epochs=100, batch_size=128, callbacks=callbacks_list)"""

# Разобраться с этого момента!!!!!
# load the network weights
filename = "weights-improvement-64-1.6889.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

int_to_char = dict((i, c) for i, c in enumerate(chars))

# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
#start = 40
pattern = dataX[start]
print("Источник:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
print("Сгенерированный текст:")
# generate characters
for i in range(chars_for_gen):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]

print("\nКонец.")