|언어|라이브러리|버전|컴포넌트/모듈|클래스|용도|
|---|---|---|---|---|---|
|Python|표준 라이브러리||sys|stdout|표준 출력 함수 사용|
||tensorflow||tensorflow.keras.datasets|mnist|MNIST 손글씨 데이터셋 활용|
||matplotlib||matplotlib|pyplot|데이터 시각화|

1. 학습셋, 테스트셋 분리
```python
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```
2. 첫번째 학습셋 이미지 확인
```python
plt.imshow(X_train[0], cmap='Greys')
plt.show()
```
3. 숫자로 출력
```python
import sys
for x in X_train[0]:
  for i in x:
    stdout.write(f"{i:3}") # 3자리 문자열로 출력 
  stdout.write('\n')
```
4. 2차원 배열을 1차원 배열로 바꿈
> [!NOTE]
> 28×28 = 784개의 속성을 이용해 0~9의 클래스 열 개 중 하나를 맞히는 문제가 됨

   ```python
   X_train = X_train.reshape(X_train.shape[0], 784)
   ```

```
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
```

- 커널(은닉층)의 수: 32
- 커널 크기: 3x3
- 입력 이미지 크기: 28x28
- 입력 이미지 색상 채널 수: 1(흑백)
- 활성화 함수: relu
