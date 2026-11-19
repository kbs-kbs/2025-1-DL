|언어|라이브러리|버전|컴포넌트/모듈|클래스|용도|
|---|---|---|---|---|---|
|Python|표준 라이브러리||sys|stdout|표준 출력 함수 사용|
||tensorflow||tensorflow.keras.datasets|mnist|MNIST 손글씨 데이터셋 활용|
||matplotlib||matplotlib|pyplot|데이터 시각화|

1. **학습셋, 테스트셋 분리**
```python
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

2. **첫번째 학습셋 이미지 확인**
```python
plt.imshow(X_train[0], cmap='Greys')
plt.show()
```

3. **숫자로 출력**
```python
import sys
for x in X_train[0]:
  for i in x:
    stdout.write(f"{i:3}") # 3자리 문자열로 출력 
  stdout.write('\n')
```

4. **2차원 배열을 1차원 배열로 바꿈**
> [!NOTE]
> 28×28 = 784개의 속성을 이용해 0~9의 클래스 열 개 중 하나를 맞히는 문제가 됨
```python
X_train = X_train.reshape(X_train.shape[0], 784)
```

5. **정규화**
   - 케라스는 데이터를 0에서 1 사이의 값으로 변환한 후 구동할 때 최적의 성능을 보임
```python
X_train = X_train.astype('float64')/255
```

6. **4, 5 합치기**
```python
X_train = X_train.reshape(X_train.shape[0], 784).astype('float64')/255
```

7. **y_train 확인**
```python
stdout.write(f"class: {y_train[0]}")
```

8. **\[5]를 \[0,0,0,0,0,1,0,0,0,0] 이러한 원-핫 인코딩 방식으로 변환**
```python
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```


```
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
```

- 커널(은닉층)의 수: 32
- 커널 크기: 3x3
- 입력 이미지 크기: 28x28
- 입력 이미지 색상 채널 수: 1(흑백)
- 활성화 함수: relu

---

## 보강 — 4주차 CNN 파트2 (맥스 풀링 · 드롭아웃 · 플래튼 · 실행)

> 교재 16장 4~5절(맥스 풀링·드롭아웃·플래튼, 컨볼루션 신경망 실행) 정리. 위에 정리한 MNIST 데이터 준비 과정에 이어지는 모델 부분입니다.

### 1. 맥스 풀링

- 컨볼루션 층으로 특징을 뽑았어도 결과가 여전히 크고 복잡하다 → **풀링(pooling, 서브 샘플링)** 으로 한 번 더 축소한다.
- 정해진 구역으로 나누고 **각 구역의 가장 큰 값만** 다음 층으로 넘긴다.
  - 예: 4×4 → 2×2 구역 네 개 → 각 구역의 최댓값 4개 → 2×2
- `MaxPooling2D(pool_size=(2,2))` — 가로 2, 세로 2 크기의 풀링 창
- 평균 풀링(`AveragePooling`)은 구역의 평균값을 쓰는 방식이며, 보편적으로는 맥스 풀링을 쓴다.

### 2. 드롭아웃 · 플래튼

- **드롭아웃(dropout)**: 은닉층 노드 중 일부를 임의로 꺼서 특정 노드에 지나치게 의존하는 **과적합을 방지**한다.
  - `Dropout(0.25)` → 25%의 노드를 계산하지 않음
- **플래튼(flatten)**: 컨볼루션·풀링 층은 이미지를 2차원 배열인 채로 다루므로, Dense 층(활성화 함수가 있는 층)에 넣기 전에 1차원으로 바꾼다.
  - `Flatten()`

### 3. 컨볼루션 신경망 실행

```python
# 컨볼루션 신경망의 설정
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

- 나머지(모델 저장 `ModelCheckpoint`, 조기 중단 `EarlyStopping`, `model.fit()`, 그래프 출력)는 16.2절 기본 프레임 코드를 그대로 쓴다.

### 4. 실행 결과

| 모델 | 베스트 에포크 | 중단 에포크 | 테스트 정확도 |
|---|---|---|---|
| 은닉층 1개 기본 프레임 | 13 | 23 | 98.16% |
| **CNN (위 구조)** | 8 | 23 | **99.17%** |

- 99.17%는 테스트 이미지 1만 장 중 9,917장을 맞혔다는 의미로, 기본 프레임보다 101장을 더 맞혔다.
- 100%가 아닌 이유는 사람도 읽기 어려운 글씨가 데이터에 있기 때문이다.
- 학습셋 오차는 계속 줄지만 테스트셋 오차가 다시 올라가기 전에 학습이 멈춘다(과적합 방지).
