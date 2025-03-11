선언: 컴파일러에게 알려줌

## 목표
진찰 기록으로 폐암 수술 환자의 생존율을 예측하는 모델 생성



## 사용 라이브러리
|언어|버전|라이브러리|버전|사용 모듈|용도|
|---|---|---|---|---|---|
|Python|3.11.11|tensorflow.keras.models|2.18.0|Sequential|전체 모델 구조를 정의|
|||tensorflow.keras.layers|2.18.0|Dense|모델 내부의 개별 층을 구성|
|||numpy|1.26.4||데이터 불러오기|



## 외부 데이터 소스 의존성
- TensorFlow: 오프라인 환경에서 GPU 가속을 사용하는 경우, CUDA와 cuDNN 라이브러리가 필요합니다. 이들은 NVIDIA 웹사이트에서 다운로드해야 하며, 설치를 위해 미리 준비해야 합니다.

- NumPy: 고성능 연산을 위해 BLAS/LAPACK 라이브러리(OpenBLAS, MKL 등)에 의존합니다. 대부분의 배포판에 포함되어 있지만, 오프라인 환경에서는 별도로 설치해야 할 수도 있습니다.

- 수술 환자 데이터: `!git clone https://github.com/taehojo/data.git`

## 단계

- model.compile(): 모델 학습 준비를 설정한다.
- model.fit(): 모델을 학습시킨다.
- model.predict(): 모델로 예측(추론)을 수행한다.


## 코드

### 모델 생성
```python
from tensorflow.keras.models import Sequential  # 텐서플로의 케라스 API에서 필요한 함수들을 불러옵니다.
from tensorflow.keras.layers import Dense       # 데이터를 다루는 데 필요한 라이브러리를 불러옵니다.
import numpy as np

!git clone https://github.com/taehojo/data.git   # 깃허브에 준비된 데이터를 가져옵니다.

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
```python
# 새로운 환자의 데이터를 입력합니다 (예: 16개의 특성을 가진 데이터)
# 실제 데이터 형식에 맞게 값을 수정해야 합니다.
new_patient_data = np.array([[45, 2, 3, 1, 0, 0, 1, 0, 0, 1, 2, 3, 1, 0, 1, 0]])

# 모델을 사용하여 예측 수행
prediction = model.predict(new_patient_data)

# 결과 출력
print("새로운 환자의 생존 확률: {:.2f}%".format(prediction[0][0] * 100))
```
