# 9주차 — 세상에 없는 얼굴, GAN (생성자 · 판별자 · 적대적 학습)

> 교재 「모두의 딥러닝」 19장 1~3절 · 9주차(GAN·오토인코더 파트1) 수업 내용 정리

## 1. GAN이란

- **생성적 적대 신경망(Generative Adversarial Networks, GAN)** 은 가짜 이미지를 만들어 내는 알고리즘이다.
- 이름의 '적대적(adversarial)'은 GAN 내부에서 벌어지는 **경합**을 뜻한다. GAN을 제안한 이안 굿펠로(Ian Goodfellow)는 **위조지폐범과 경찰**의 예를 들었다.
  - 위조지폐범(생성자)은 진짜와 똑같은 가짜를 만들려 하고,
  - 경찰(판별자)은 가짜를 가려내려 한다.
  - 이 경합이 반복되면 결국 **더 정교한 가짜**가 만들어진다.
- 구성 요소
  - **생성자(Generator)**: 가짜를 만들어 내는 파트
  - **판별자(Discriminator)**: 진위를 가려내는 파트
- 대표 변형인 **DCGAN(Deep Convolutional GAN)** 은 컨볼루션 신경망을 GAN에 적용해, 불안정하던 초기 GAN을 크게 보완했다. 이 장의 실습도 DCGAN이다.

## 2. 생성자(Generator)

- 랜덤한 픽셀 값으로 시작해, 판별자의 판별 결과에 따라 업데이트되며 점차 원하는 이미지를 만든다.
- 일반 CNN과 다른 점
  - 옵티마이저/컴파일 과정이 **없다** (학습·판별은 판별자 쪽에서 일어난다).
  - 풀링 대신 **패딩(padding='same')** 을 쓴다. 컨볼루션을 지나면 크기가 줄어드는데, 생성 결과는 판별자가 비교할 '진짜'와 **같은 크기(예: MNIST 28×28)** 여야 하기 때문이다.
  - **배치 정규화(BatchNormalization)** 로 층이 늘어나도 안정적으로 학습한다.
  - 활성화 함수는 **ReLU 계열**, 판별자로 넘기기 직전에는 **tanh** 를 써서 출력 값을 -1~1로 맞춘다(판별자에 들어가는 MNIST 픽셀도 -1~1로 맞춘다).

```python
generator = Sequential()
generator.add(Dense(128*7*7, input_dim=100, activation=LeakyReLU(0.2)))  # ① 100차원 랜덤 벡터
generator.add(BatchNormalization())                                      # ②
generator.add(Reshape((7, 7, 128)))                                      # ③ 7×7×128 이미지로 재배열
generator.add(UpSampling2D())                                            # ④ 7→14
generator.add(Conv2D(64, kernel_size=5, padding='same'))                 # ⑤
generator.add(BatchNormalization())                                      # ⑥
generator.add(Activation(LeakyReLU(0.2)))                                # ⑦
generator.add(UpSampling2D())                                            # ⑧ 14→28
generator.add(Conv2D(1, kernel_size=5, padding='same', activation='tanh'))  # ⑨ 28×28×1
```

- `input_dim=100`은 100차원 크기의 랜덤 벡터를 넣는다는 의미다(100이 아니어도 된다).
- `UpSampling2D()`는 가로·세로 크기를 **두 배씩** 늘린다. 7×7 → 14×14 → 28×28.
- 케라스에는 UpSampling과 Conv2D를 합친 `Conv2DTranspose()`도 있지만, 원리를 이해하기 위해 두 개로 나누어 쓴다.
- **LeakyReLU(0.2)**: ReLU는 x가 음수면 무조건 0이 되어 뉴런이 일찍 소실되는데, 0 이하에서도 작은 값(0.2배)을 갖게 해 학습을 안정시키는 함수다. GAN에서는 일반 ReLU를 쓰면 학습이 불안정해지는 경우가 많다.

## 3. 판별자(Discriminator)

- 컨볼루션 신경망 구조를 그대로 가져와 "진짜(1) / 가짜(0)"를 판별한다. 이진 분류이므로 `binary_crossentropy` + `adam`.
- `strides=2`는 커널 윈도를 두 칸씩 이동시켜, 드롭아웃·풀링처럼 크기를 줄이고 새로운 특징을 뽑는 효과를 낸다.
- **주의**: 판별자는 판별만 하고 **자기 자신은 학습하지 않는다.** 얻은 가중치는 생성자를 업데이트하는 데 쓰이므로 `discriminator.trainable = False`로 학습 기능을 꺼 준다.

```python
discriminator = Sequential()
discriminator.add(Conv2D(64, kernel_size=5, strides=2, input_shape=(28,28,1), padding='same'))
discriminator.add(Activation(LeakyReLU(0.2)))
discriminator.add(Dropout(0.3))
discriminator.add(Conv2D(128, kernel_size=5, strides=2, padding='same'))
discriminator.add(Activation(LeakyReLU(0.2)))
discriminator.add(Dropout(0.3))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer='adam')
discriminator.trainable = False
```

## 4. 적대적 신경망 실행

- 생성자와 판별자를 **연결**한다는 것은 생성자의 출력을 판별자에 넣어 진위를 판별하게 만든다는 뜻이다.
  - 생성자 `G()`, 판별자 `D()`, 실제 데이터 `x`, 입력 `input`이라 하면
  - 생성자는 `D(G(input))`이 참(1)이라고 주장하고, 판별자는 `D(x)`만 참이라고 여긴다.
  - 학습이 진행되어 **판별자가 둘을 구별하지 못해 정확도가 0.5에 가까워지면** 생성자는 역할을 다한 것이고 학습을 종료한다.

```python
# 생성자와 판별자를 연결한 gan 모델
ginput = Input(shape=(100,))
dis_output = discriminator(generator(ginput))
gan = Model(ginput, dis_output)
gan.compile(loss='binary_crossentropy', optimizer='adam')
gan.summary()
```

```python
def gan_train(epoch, batch_size, saving_interval):
    (X_train, _), (_, _) = mnist.load_data()          # 테스트 과정은 필요 없음
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_train = (X_train - 127.5) / 127.5               # 0~255 → -1~1 (tanh 출력과 범위 일치)

    for i in range(epoch):
        # ① 진짜 이미지를 판별자에 넣고 '모두 참(1)' 레이블로 한 번 학습
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        true = np.ones((batch_size, 1))
        d_loss_real = discriminator.train_on_batch(imgs, true)

        # ② 생성자가 만든 가짜 이미지를 '모두 거짓(0)' 레이블로 학습
        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(noise)
        fake = np.zeros((batch_size, 1))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)

        # ③ 생성자는 판별자를 속이는 방향으로 학습 (레이블은 '참(1)')
        g_loss = gan.train_on_batch(noise, true)
        ...
```

- `train_on_batch(x, y)`는 입력과 레이블을 받아 **딱 한 번** 학습시켜 모델을 업데이트한다.
- 판별자 오차는 실제/가짜 두 번의 평균(`(d_loss_real + d_loss_fake) / 2`)으로 본다.
- `saving_interval`마다 `generator.predict()`로 25장(5×5)을 만들어 `gan_images` 폴더에 저장한다. (저장 코드는 이 장의 주된 목표가 아니어서 소스 코드로만 소개)

## 5. 실행 결과

- Epoch 2000까지 진행하면 점점 숫자에 가까운 형태가 만들어진다.
- 생성된 이미지는 `gan_images` 폴더에 저장된다.

## 6. 팁 정리

| 항목 | 내용 |
|---|---|
| 생성자 활성화 함수 | LeakyReLU(0.2), 마지막은 tanh(-1~1) |
| 판별자 활성화 함수 | LeakyReLU(0.2), 마지막은 sigmoid(0/1) |
| 패딩 | 생성자는 `padding='same'`으로 크기 유지 |
| 학습 순서 | 판별자(진짜) → 판별자(가짜) → 생성자(gan 모델) |
| 종료 시점 | 판별자 정확도가 0.5에 가까워질 때 |
