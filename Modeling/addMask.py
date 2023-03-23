import tensorflow as tf
from keras.layers import Input, Embedding, Masking, LSTM, Dense
from keras.layers.attention import Attention  # Assuming you have implemented the Attention layer separately

class Seq2Seq_mask(tf.keras.Model):
    def __init__(self, kor_vocab_size, eng_vocab_size, latent_dim, max_length_kor, max_length_eng):
        super(Seq2Seq_mask, self).__init__()
        self.encoder_inputs = Input(shape=(max_length_kor,))
        self.encoder_embed = Embedding(kor_vocab_size, latent_dim, mask_zero=True)(self.encoder_inputs)
        self.encoder_mask = Masking(mask_value=0.)(self.encoder_embed)
        self.encoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True)
        self.encoder_output1, self.state_h1, self.state_c1 = self.encoder_lstm1(self.encoder_mask)
        self.encoder_lstm2 = LSTM(latent_dim, return_sequences=True, return_state=True)
        self.encoder_output, self.state_h, self.state_c = self.encoder_lstm2(self.encoder_output1)

        self.decoder_inputs = Input(shape=(max_length_eng,))
        self.decoder_embed = Embedding(eng_vocab_size, latent_dim, mask_zero=True)(self.decoder_inputs)
        self.decoder_mask = Masking(mask_value=0.)(self.decoder_embed)
        self.decoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True)
        self.decoder_output1, _, _ = self.decoder_lstm1(self.decoder_mask, initial_state=[self.state_h1, self.state_c1])
        self.decoder_lstm2 = LSTM(latent_dim, return_sequences=True, return_state=True)
        self.decoder_output, _, _ = self.decoder_lstm2(self.decoder_output1, initial_state=[self.state_h, self.state_c])

        self.attention = Attention()([self.decoder_output, self.encoder_output])
        self.concat = tf.keras.layers.concatenate([self.decoder_output, self.attention], axis=-1)
        self.decoder_dense = Dense(eng_vocab_size, activation='softmax')(self.concat)

        self.model = tf.keras.Model([self.encoder_inputs, self.decoder_inputs], self.decoder_dense)

    def call(self, inputs):
        encoder_inputs, decoder_inputs = inputs
        encoder_embed = self.encoder_embed(encoder_inputs)
        encoder_mask = self.encoder_mask(encoder_embed)
        encoder_output1, state_h1, state_c1 = self.encoder_lstm1(encoder_mask)
        encoder_output, state_h, state_c = self.encoder_lstm2(encoder_output1)

        decoder_embed = self.decoder_embed(decoder_inputs)
        decoder_mask = self.decoder_mask(decoder_embed)
        decoder_output1, _, _ = self.decoder_lstm1(decoder_mask, initial_state=[state_h1, state_c1])
        decoder_output, _, _ = self.decoder_lstm2(decoder_output1, initial_state=[state_h, state_c])

        attention = self.attention([decoder_output, encoder_output])
        concat = tf.keras.layers.concatenate([decoder_output, attention], axis=-1)
        decoder_dense = self.decoder_dense(concat)

        return decoder_dense
