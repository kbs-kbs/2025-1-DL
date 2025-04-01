|언어|라이브러리|버전|컴포넌트/모듈|클래스|용도|
|---|---|---|---|---|---|
|Python|tensorflow||tensorflow.keras.datasets|mnist|MNIST 손글씨 데이터셋 활용|



```
from tensorflow.keras.datasets import mnist
```





```
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
```

- 커널(은닉층)의 수: 32
- 커널 크기: 3x3
- 입력 이미지 크기: 28x28
- 입력 이미지 색상 채널 수: 1(흑백)
- 활성화 함수: relu
