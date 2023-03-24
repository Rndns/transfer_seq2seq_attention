# transfer_seq2seq_attention

## 프영 seq2seq 모델
### 학습 모델
![트레인디코너모델](https://user-images.githubusercontent.com/95123300/227431159-da7bed92-16e8-42de-8e7c-fc9231ffc4fa.png)

### 인코더 모델
![인코더모델](https://user-images.githubusercontent.com/95123300/227431205-59a27ab7-c00e-4f99-87df-9073aa1a55ff.png)

### 디코더 모델
![디코더모델](https://user-images.githubusercontent.com/95123300/227431221-abe9bcae-3c97-42c1-9c7d-bae542c7569a.png)

### 평가
![bleu값](https://user-images.githubusercontent.com/95123300/227431310-21b88c3a-4fb5-49ca-a8b4-dbbb8b7864f5.png)

----------------------------------------------------------------------------------------------------------------------------

## 한영 seq2seq 
### 모델 
![image](https://user-images.githubusercontent.com/95123300/227430139-d3f0dad4-5534-4be8-b32d-d654b76bc6db.png)

### 모듈화
#### mode
- argument를 이용한 train mode와 predict mode로 나누어 실행
<img width="322" alt="image" src="https://user-images.githubusercontent.com/95123300/227432514-fcca1f32-fffc-4497-a3fe-51222335a2f0.png">

#### 구조
- Data: input data, test data
- Model: incoder, decoder
- Modeling: modularization seq2seq, incoder, decoder / inference
- Proprecessing: prorecessing
- app.py: main
- config.ini: save config
<img width="146" alt="image" src="https://user-images.githubusercontent.com/95123300/227433894-865ca657-baa7-4cb3-9e4f-fb64dc5e70e2.png">

#### config
- import configparser를 이용한 설정 저장
![image](https://user-images.githubusercontent.com/95123300/227433570-ef0cbe79-e402-4552-8adb-a61f629b2c68.png)
