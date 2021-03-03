from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재밌어요', '참 최고예요', '참 잘 만든 영화예요', '추천하고 싶은 영화입니다',
        '한 번 더 보고 싶네요', '글쎄요', '별로예요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밌네요', '규현이가 잘 생기긴 했어요']

# 긍정1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x)

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding = 'pre')
print(pad_x)
print(pad_x.shape) # (13, 5)
pad_x = pad_x.reshape(pad_x.shape[0],pad_x.shape[1],1)

print(np.unique(pad_x)) # 하나씩만 추출해서 리스트화 - 11이 없다!
print(len(np.unique(pad_x))) # 27



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D

model = Sequential()
# model.add(Embedding(input_dim = 28, output_dim=11, input_length=5))
# model.add(Embedding(28,11)) # Flatten 하려면 input_length 를 줘야한다!
model.add(LSTM(128, activation = 'relu', input_shape = (5,1)))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(1, activation='sigmoid'))
# model.add(Dense(1))

model.summary()

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])
model.fit(pad_x, labels, epochs = 150)

acc = model.evaluate(pad_x, labels)[1]
print(acc)

# Dense
# 0.9230769276618958

# LSTM
# 0.9230769276618958