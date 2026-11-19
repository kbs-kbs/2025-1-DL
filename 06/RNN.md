- w(window size): 윈도우 크기, 이전 데이터 몇 개를 묶을 것인지
- h(horizon factor): 예측할 미래 값이 윈도우로부터 떨어진 정도, 얼마나 먼 미래 값을 예측할 것인지.

---

## 보강 — 6주차 시퀀스 배열로 다루는 순환 신경망(RNN)

> 교재 18장 1~3절 정리 · 위의 w/h 메모는 시계열 예측(윈도우·호라이즌) 용어로, 아래 LSTM·어텐션 내용과 함께 보면 된다.

### 1. RNN이 필요한 이유

- 지금까지의 데이터는 순서와 관계없이 입력됐지만, **문장은 단어의 순서가 의미를 결정**한다.
- **순환 신경망(RNN)** 은 입력을 순서대로 받아 **앞서 입력된 데이터를 잠시 기억**해 둔다. 기억된 값이 얼마나 중요한지 판단해 가중치를 주어 다음 입력으로 넘긴다.
- 같은 층 안을 맴도는 것처럼 보여 ‘순환’ 신경망이라 부른다.
- “오늘 주가가 몇이야?”와 “어제 주가가 몇이야?”는 둘 다 ‘주가’를 포함하지만, 앞 단어에 따라 다른 결과가 나와야 한다 → RNN이 필요한 이유.

### 2. LSTM

- RNN은 한 층 안에서 반복이 많아 **기울기 소실**이 잘 생긴다.
- **LSTM(Long Short Term Memory)** 은 반복 직전에 **기억 값을 다음 층으로 넘길지 관리하는 단계(게이트)** 를 하나 더 두어 이를 보완한다.
- LSTM의 활성화 함수로는 주로 **tanh** 를 쓴다(`activation='tanh'`).

### 3. 실습 ① 로이터 뉴스 카테고리 분류

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# 데이터를 불러와 학습셋, 테스트셋으로 나눕니다.
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=1000, test_split=0.2)

category = np.max(y_train) + 1        # 46개 카테고리

# 단어의 수를 맞춥니다.
X_train = sequence.pad_sequences(X_train, maxlen=100)
X_test  = sequence.pad_sequences(X_test,  maxlen=100)

y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)

model = Sequential()
model.add(Embedding(1000, 100))
model.add(LSTM(100, activation='tanh'))
model.add(Dense(46, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(X_train, y_train, batch_size=20, epochs=200,
                    validation_data=(X_test, y_test), callbacks=[early_stopping_callback])
```

- 로이터 데이터: 11,228개 기사 / 46개 카테고리 / 학습 8,982 + 테스트 2,246
- `num_words=1000` = 빈도 1~1,000위 단어만 사용
- `maxlen=100` = 기사 단어 수를 100개로 맞춤(넘으면 자르고, 모자라면 0으로 채움)
- 기사는 단어가 아니라 **빈도 순번 숫자**로 들어 있다(예: 3 = 세 번째로 빈도가 높은 단어).
- 결과: **16번째 에포크에서 자동 중단, 정확도 71.7%**

### 4. 실습 ② LSTM + CNN으로 영화 리뷰 분류

```python
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, LSTM, Conv1D, MaxPooling1D
from tensorflow.keras.datasets import imdb

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=5000)
X_train = sequence.pad_sequences(X_train, maxlen=500)
X_test  = sequence.pad_sequences(X_test,  maxlen=500)

model = Sequential()
model.add(Embedding(5000, 100))
model.add(Dropout(0.5))
model.add(Conv1D(64, 5, padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(55))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

- IMDB: 영화 리뷰 2만 5,000여 개, 긍정/부정 **2클래스**라서 원-핫 인코딩 과정이 없다.
- **Conv1D / MaxPooling1D**: 2차원 이미지용 Conv2D·MaxPooling2D를 **1차원 배열**로 옮긴 것. 커널이 1차원 배열 위를 지나가며 새 층을 만든다.
- 결과: **5번째 에포크에서 자동 중단, 정확도 87.96%** (중단 시점은 실행마다 달라질 수 있음)

### 5. 더 나아가기

- 어텐션(7주차)은 같은 IMDB 모델에 `Attention()` 층을 넣어 정확도를 88.08%까지 올렸다 → `07/어텐션.md`
