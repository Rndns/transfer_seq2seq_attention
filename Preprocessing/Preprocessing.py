import re
import pandas as pd

from konlpy.tag import Okt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class Preprocess:
    Data01_path = '/Users/seongjinkim/Documents/workspace/project/translator/Data/구어체01.xlsx'

    def __init__(self, NUM_SAMPLES = 200000, Method = 'post', Test_set = 0.1):
        self.NUM_SAMPLES = NUM_SAMPLES
        self.Method = Method
        self.Test_set = Test_set


    def preprocessing(self):
        kor_corpus, eng_corpus = self.readData()

        kor_corpus = self.filter_kor(kor_corpus)
        eng_corpus = self.filter_eng(eng_corpus)
        sents_kor_in, sents_en_in, sents_en_out = self.split_data(kor_corpus, eng_corpus)
        encoder_input, decoder_input, decoder_target, kor_vocab_size, eng_vocab_size, tokenizer_enc, tokenizer_dec = self.tokenizer(sents_kor_in, sents_en_in, sents_en_out)
        encoder_input, decoder_input, decoder_target = self.padding(encoder_input, decoder_input, decoder_target)

        max_length_kor, max_length_eng = encoder_input.shape[1], decoder_input.shape[1]

        return encoder_input, decoder_input, decoder_target, kor_vocab_size, eng_vocab_size, max_length_kor, max_length_eng, tokenizer_enc, tokenizer_dec


    def read_data(self):
        df = pd.read_excel(Preprocess.Data01_path)

        kor_corpus, eng_corpus = [], []

        for kor, eng in zip(df.원문, df.번역문):
            kor_corpus.append(kor)
            eng_corpus.append(eng)

        return kor_corpus, eng_corpus


    def filter_kor(self, text):
        okt = Okt()

        text = re.sub(r"([?.!,¿])", r" \1", text)
        text = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣ!.?]+", r" ", text)
        sent = re.sub(r"\s+", " ", sent)
        text = ' '.join(okt.morphs(text))

        return text


    def filter_eng(self, text):
        text = text.lower()
        text = re.sub(r"([?.!,¿])", r" \1", text)
        text = re.sub(r"[^a-zA-Z!.?]+", r" ", text)
        sent = re.sub(r"\s+", " ", sent)

        return text


    def split_data(self, kor_corpus, eng_corpus):
        sents_kor_in, sents_en_in, sents_en_out = [], [], []

        for i, (src_line, tar_line) in enumerate(zip(kor_corpus, eng_corpus)):

            src_line = [w for w in self.preprocess_kor(src_line).split()]

            tar_line = self.preprocess_eng(tar_line)
            tar_line_in = [w for w in ("<sos> " + tar_line).split()]
            tar_line_out = [w for w in (tar_line + " <eos>").split()]

            sents_kor_in.append(src_line)
            sents_en_in.append(tar_line_in)
            sents_en_out.append(tar_line_out)
        
            if i == self.NUM_SAMPLES - 1:
                break

        return sents_kor_in, sents_en_in, sents_en_out


    def tokenizer(self, sents_kor_in, sents_en_in, sents_en_out):
        tokenizer_enc = Tokenizer(filters="", lower=False)
        tokenizer_enc.fit_on_texts(sents_kor_in)

        encoder_input = tokenizer_enc.texts_to_sequences(sents_kor_in)

        tokenizer_dec = Tokenizer(filters="", lower=False)
        tokenizer_dec.fit_on_texts(sents_en_in)
        tokenizer_dec.fit_on_texts(sents_en_out)

        decoder_input = tokenizer_dec.texts_to_sequences(sents_en_in)
        decoder_target = tokenizer_dec.texts_to_sequences(sents_en_out)

        kor_vocab_size = len(tokenizer_enc.word_index) + 1
        eng_vocab_size = len(tokenizer_dec.word_index) + 1

        return encoder_input, decoder_input, decoder_target, kor_vocab_size, eng_vocab_size, tokenizer_enc, tokenizer_dec


    def padding(self, encoder_input, decoder_input, decoder_target):
        encoder_input = pad_sequences(encoder_input, padding = self.Method)
        decoder_input = pad_sequences(decoder_input, padding = self.Method)
        decoder_target = pad_sequences(decoder_target, padding = self.Method)
        
        return encoder_input, decoder_input, decoder_target


    def split_train_test(self, encoder_input, decoder_input, decoder_target):
        n_of_val = int(self.NUM_SAMPLES*self.Test_set)

        encoder_input_train = encoder_input[:-n_of_val] 
        decoder_input_train = decoder_input[:-n_of_val]
        decoder_target_train = decoder_target[:-n_of_val]

        encoder_input_test = encoder_input[-n_of_val:]
        decoder_input_test = decoder_input[-n_of_val:]
        decoder_target_test = decoder_target[-n_of_val:]

        return encoder_input_train, decoder_input_train, decoder_target_train, encoder_input_test, decoder_input_test, decoder_target_test
