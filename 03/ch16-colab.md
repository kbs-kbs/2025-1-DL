## 목표
진찰 기록으로 폐암 수술 환자의 생존율을 예측하는 모델 생성

## 사용 라이브러리
|언어|버전|라이브러리|버전|모듈/클래스|용도|
|---|---|---|---|---|---|
|Python|3.11.11|tensorflow|2.18.0|keras.models.Sequential|전체 모델 구조를 정의|
|||||keras.layers.Dense|모델 내부의 개별 층을 구성|
|||numpy|1.26.4||데이터 불러오기|



## 외부 리소스 의존성
|라이브러리|외부 리소스|용도|
|---|---|---|
|tensorflow|CUDA, cuDNN|GPU 가속|
|수술 환자 데이터|https://github.com/taehojo/data.git||

## 단계

- model.compile(): 모델 학습 준비를 설정한다.
- model.fit(): 모델을 학습시킨다.
- model.predict(): 모델로 예측(추론)을 수행한다.


## 코드

### 모델 생성
```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
import numpy as np
import os

# MNIST 데이터를 불러옵니다. 
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 차원 변환 후, 테스트셋과 학습셋으로 나누어 줍니다.
X_train = X_train.reshape(X_train.shape[0], 784).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 784).astype('float32') / 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 모델 구조를 설정합니다.
model = Sequential()
model.add(Dense(512, input_dim=784, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()
```
