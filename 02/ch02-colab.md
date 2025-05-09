선언: 컴파일러에게 알려줌

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
```bash
git clone https://github.com/taehojo/data.git
```

```python title:ch02-colab.ipynb
from tensorflow.keras.models import Sequential  # 텐서플로의 케라스 API에서 필요한 함수들을 불러옵니다.
from tensorflow.keras.layers import Dense       # 데이터를 다루는 데 필요한 라이브러리를 불러옵니다.
import numpy as np

Data_set = np.loadtxt("./data/ThoraricSurgery3.csv", delimiter=",")  # 수술 환자 데이터를 불러옵니다.
X = Data_set[:,0:16]                                                 # 환자의 진찰 기록을 X로 지정합니다.
y = Data_set[:,16]                                                   # 수술 후 사망/생존 여부를 y로 지정합니다.

model = Sequential()                                                  # 딥러닝 모델의 구조를 결정합니다.
model.add(Dense(30, input_dim=16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # 딥러닝 모델을 실행합니다.
history=model.fit(X, y, epochs=5, batch_size=16)

```

### 모델 사용
```python title:ch02-colab.ipynb
# 새로운 환자 데이터 예측
new_patient_data = np.array([[45, 2, 3, 1, 0, 0, 1, 0, 0, 1, 2, 3, 1, 0, 1, 0]])
prediction_prob = model.predict(new_patient_data)

# 확률을 이항 분류 결과로 변환 (0.5를 임계값으로 사용)
prediction_class = (prediction_prob > 0.5).astype(int)

print("예측 확률: {:.2f}%".format(prediction_prob[0][0] * 100))
print("이항 분류 결과:", "생존" if prediction_class[0][0] == 1 else "사망")
```

prediction_prob > 0.5:

이 부분은 prediction_prob 배열의 각 요소를 0.5와 비교합니다.

결과는 불리언(Boolean) 배열로, 각 요소가 0.5보다 크면 True, 그렇지 않으면 False입니다.

.astype(int):

이 메소드는 불리언 배열을 정수 배열로 변환합니다.

True는 1로, False는 0으로 변환됩니다.
