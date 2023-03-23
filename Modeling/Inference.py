import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

class Inference:
    def __init__(self) -> None:
        pass
    
    def inference(model, input_sequence_list, kor_tokenizer, eng_tokenizer, max_length_kor, max_length_eng):
        predicted_sentences = []
        for input_sequence in input_sequence_list:
            # input_sequence = kor_tokenizer.texts_to_sequences([input_sentence])
            # input_sequence = pad_sequences(input_sequence, maxlen=max_length_kor, padding='post')
            
            encoder_output, state_h1, state_c1 = model.layers[2](model.layers[1](input_sequence))
            encoder_output, state_h, state_c = model.layers[4](encoder_output)

            decoder_input = tf.expand_dims([eng_tokenizer.word_index['<sos>']], 0)
            decoder_output = []

            while True:
                decoder_output1, state_h1, state_c1 = model.layers[6](model.layers[5](decoder_input), initial_state=[state_h1, state_c1])
                decoder_output2, state_h, state_c = model.layers[8](decoder_output1, initial_state=[state_h, state_c])
                attention = model.layers[9]([decoder_output2, encoder_output])
                concat = model.layers[10]([decoder_output2, attention])
                decoder_dense = model.layers[11](concat)

                predicted_id = tf.argmax(decoder_dense, axis=-1).numpy()[0][0]

                if eng_tokenizer.index_word[predicted_id] == '<end>' or len(decoder_output) >= max_length_eng:
                    break

                decoder_output.append(predicted_id)

                decoder_input = tf.expand_dims([predicted_id], 0)

            predicted_sentence = ' '.join(eng_tokenizer.index_word[idx] for idx in decoder_output)
            predicted_sentences.append(predicted_sentence)

        return predicted_sentences