# transfer_seq2seq_attention

## 프영 seq2seq 모델
### 학습 모델
![성공한 코드](https://user-images.githubusercontent.com/95123300/227435748-a433dc26-8e40-4ad4-913c-a67b55d01f41.png)
![트레인디코너모델](https://user-images.githubusercontent.com/95123300/227431159-da7bed92-16e8-42de-8e7c-fc9231ffc4fa.png)

### 인코더 모델
![인코더모델](https://user-images.githubusercontent.com/95123300/227431205-59a27ab7-c00e-4f99-87df-9073aa1a55ff.png)

### 디코더 모델
![image](https://user-images.githubusercontent.com/95123300/227438053-15ee4500-753f-48de-b59d-4d1aab695a16.png)
![디코더모델](https://user-images.githubusercontent.com/95123300/227431221-abe9bcae-3c97-42c1-9c7d-bae542c7569a.png)

### 평가
![bleu값](https://user-images.githubusercontent.com/95123300/227431310-21b88c3a-4fb5-49ca-a8b4-dbbb8b7864f5.png)

### 과정
![성공한 코드](https://user-images.githubusercontent.com/95123300/227436053-bf7466b4-2942-4065-85dd-6431fb5ad57f.png)

### 총평
- luong의 방식으로 attention을 seq2seq에 접목시켰다.
- 인코더(트레인)lstm의 return_sequences = True로 줘서 attention_vector를 넣을 조건을 충족시킴
- 디코더(트레인)에서 Attention레이어와 concat한 attention vector가 지나갈 denselayer를 변수로 지정해줬습니다. 그 이유는 실제로 예측 값을 뽑아낼 모델에서 쓸려고 이렇게 안하면 밑에서 쓸 수가 없습니다. 다시 말해 재사용을 위해서 변수를 지정해줬습니다. 
- encoder_outputs, state_h, state_c =  encoder_lstm(enc_emb) 은닉상태를 알려주려고 이렇게 했습니다.
- encoder_model = Model(encoder_inputs, [encoder_outputs, encoder_states]) encoder_model모델의 
- 모든 은닉상태를 파악할려고 encoder_outputs추가했음
- attention_key를 통해 inputnode가 모델에게 데이터가 어떻게 들어갈건지 알려주는겁니다

----------------------------------------------------------------------------------------------------------------------------

## 한영 seq2seq 
### 모델 
![image](https://user-images.githubusercontent.com/95123300/227430139-d3f0dad4-5534-4be8-b32d-d654b76bc6db.png)

### 코드
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

#### Config
- configparser를 이용한 설정 저장과 불러오기
![image](https://user-images.githubusercontent.com/95123300/227433570-ef0cbe79-e402-4552-8adb-a61f629b2c68.png)
<img width="377" alt="image" src="https://user-images.githubusercontent.com/95123300/227434982-72e90e3e-8d0a-4495-93ac-61a0cdb33a40.png">

#### Model Modulariztion
- seq2seq
![image](https://user-images.githubusercontent.com/95123300/227437375-05c2d9f1-57cf-41b1-9749-0df0dcc6253a.png)

- encoder, decoder
<img width="867" alt="image" src="https://user-images.githubusercontent.com/95123300/227437311-c3cec1d7-094d-442d-9dd4-326ca64bb30a.png">





