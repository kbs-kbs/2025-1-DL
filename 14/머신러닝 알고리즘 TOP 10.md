# 14주차 — 심화학습 및 전체 강의내용 요약

> 교재 「모두의 딥러닝」 부록 1장(가장 많이 사용하는 머신 러닝 알고리즘 TOP 10) + `[심화학습 자료] 실전! TensorFlow로 배우는 딥러닝` 교안 · 14주차 수업 내용 정리

## Part 1. 가장 많이 사용하는 머신 러닝 알고리즘 TOP 10

### 왜 딥러닝 말고 다른 알고리즘을 배우는가

- 딥러닝은 **많은 데이터**가 필요하고, 좋은 사양의 컴퓨터와 긴 학습 시간이 든다.
- 데이터가 충분하지 않거나 빠른 결과가 필요하면 **간단한 머신 러닝 기법이 더 바람직**할 수 있다.
- 스케일링은 `StandardScaler()`. 딥러닝은 가중치 조절로 불필요한 속성을 스스로 제거하지만, 일반 머신 러닝은 두드러진 값이 결과에 큰 영향을 주므로 **전처리가 특히 중요**하다.
- 실습 데이터: 피마 인디언 당뇨병 데이터셋, 학습:테스트 = 75:25, 10겹 교차 검증(`StratifiedKFold`).

```python
# 사이킷런 기본 순서: ① 알고리즘 불러오기 → ② fit() 학습 → ③ predict() 예측
```

| # | 알고리즘 | 핵심 아이디어 | 사이킷런 |
|---|---|---|---|
| 1 | 결정 트리 (Decision Tree) | 예/아니요 질문을 계속 던져 분류. 단순해 과적합에 취약 | `sklearn.tree.DecisionTreeClassifier` |
| 2 | 랜덤 포레스트 (Random Forest) | 결정 트리를 여러 개 묶어 결과를 취합 → 과적합 보완(앙상블) | `sklearn.ensemble.RandomForestClassifier(n_estimators=50)` |
| 3 | 가우시안 나이브 베이즈 | 연속형 속성에서 베이즈 정리로 각 범주의 확률 계산. 속성은 상호 독립 가정 | `sklearn.naive_bayes.GaussianNB` |
| 4 | k-최근접 이웃 (k-NN) | 새 데이터에서 가장 가까운 k개를 보고 종류 결정. k 값 선택이 관건 | `sklearn.neighbors.KNeighborsClassifier` |
| 5 | 에이다 부스트 (AdaBoost) | 틀린 데이터에 가중치를 주며 반복 분류 → 약한 분류기를 강하게(부스팅) | `sklearn.ensemble.AdaBoostClassifier` |
| 6 | 이차 판별 분석 (QDA) | 클래스 분포의 중심과 모양으로 경계를 그림. 곡선 경계(비선형) 가능, 데이터·연산 더 필요 | `sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis` |
| 7 | 서포트 벡터 머신 (SVM) | 서포트 벡터와의 거리(마진)를 최대화하는 경계 | `sklearn.svm.SVC(kernel='linear')` |
| 8 | SVM - RBF 커널 | 선형 분리가 안 될 때 데이터를 고차원으로 옮겨 분리 | `SVC(kernel='rbf')` |
| 9 | 보팅 (Voting) | **서로 다른** 알고리즘 여러 개를 같은 데이터에 적용해 다수결/평균 | `sklearn.ensemble.VotingClassifier` |
| 10 | 배깅 (Bagging) | **하나의** 알고리즘을 부트스트랩으로 만든 여러 데이터셋에 각각 학습 후 취합 | `BaggingClassifier(base_estimator=SVC(kernel='rbf'), n_estimators=10)` |

### 앙상블 세 가지 구분

| 방법 | 데이터 | 분류기 | 연결 방식 |
|---|---|---|---|
| 보팅 | 같은 데이터셋 | 서로 **다른** 알고리즘 | 결과를 모아 다수결/평균 |
| 배깅 | 데이터셋을 **여러 개로 분리**(부트스트랩) | 같은 알고리즘 | 각자 따로 학습 후 취합 |
| 부스팅 | 같은 데이터(가중치 변경) | 같은 알고리즘 | 이전 분류기의 오류를 다음 분류기가 보완 |

### 성능 비교

- 열 개 알고리즘을 한 번에 실행해 비교하면, 이 데이터에서는 **에이다 부스트가 75.88%** 로 가장 높았다.
- 교차 검증마다 데이터 조합이 달라져 결과는 실행할 때마다 조금씩 달라진다(데이터가 많을수록 차이가 줄어든다).

## Part 2. [심화학습 자료] 실전! TensorFlow로 배우는 딥러닝 — 전체 목차

366쪽 교안으로, 1학기 전체 내용을 한 번에 훑는 요약 자료다.

| 장 | 주제 |
|---|---|
| 1 | Artificial Intelligence (정의·역사·튜링 테스트·응용과 이슈) |
| 2 | Machine Learning (지도/비지도 학습, 손실·비용 함수, 경사 하강법) |
| 3 | Model Validation (모델 검증) |
| 4 | Regression Analysis (회귀 분석) |
| 5 | Logistic Regression (로지스틱 회귀) |
| 6 | Artificial Neural Network (ANN) |
| 7 | Error Backpropagation & Optimization Method (오차 역전파·최적화) |
| 8 | TensorFlow (텐서플로 사용법, Colab) |
| 9 | Deep Neural Network (DNN) |
| 10 | Convolutional Neural Network (CNN) |
| 11 | Recurrent Neural Network (RNN) |
| 12 | Long Short Term Memory (LSTM) |

### 1학기 흐름 정리

1. **기본기**: 데이터 파악 → 결측치·카테고리 처리 → 회귀/분류(선형 회귀, 로지스틱 회귀, SVM, 결정 트리)
2. **신경망**: 퍼셉트론 → 다층 퍼셉트론 → 오차 역전파 → 활성화 함수(ReLU) → 옵티마이저(SGD·모멘텀·Adam)
3. **딥러닝 활용**: CNN(이미지) → NLP(토큰화·원핫·임베딩) → RNN/LSTM → 어텐션 → GAN/오토인코더 → 전이 학습 → XAI → 캐글
4. **평가**: 과적합 방지(검증셋, k겹 교차 검증, 드롭아웃, 조기 중단), 정확도·손실 곡선 해석
