import argparse
import evaluate 
import tensorflow as tf

import Preprocessing.Preprocessing as Preprocessing
import Modeling.Modeling01 as Model01
import Modeling.Inference as Inference


def get_args():
    parser = argparse.ArgumentParser()


if __name__ == '__main__':
    BATCH_SIZE = 40000
    EPOCHS = 100
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=3)

    prepro = Preprocessing.Preprocess(200000, 'post', 0.1)
    infer = Inference.Inference()

    # preprocessing
    encoder_input, decoder_input, decoder_target, kor_vocab_size, eng_vocab_size, max_length_kor, max_length_eng, tokenizer_enc, tokenizer_dec = prepro.preprocessing()

    # split train & test
    encoder_input_train, decoder_input_train, decoder_target_train, encoder_input_test, decoder_input_test, decoder_target_test = prepro.split_train_test(encoder_input, decoder_input, decoder_target)

    # model
    model01 = Model01.Seq2Seq(kor_vocab_size, eng_vocab_size, max_length_kor, max_length_eng)

    # compile
    model01.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

    # fit
    history = model01.fit(x = [encoder_input_train, decoder_input_train], y = decoder_target_train, 
                        validation_data = ([encoder_input_test, decoder_input_test], decoder_target_test), 
                        batch_size = BATCH_SIZE, callbacks = [earlystopping], epochs = EPOCHS)
    
    # inference
    predictions = infer.inference(model01, encoder_input_test, tokenizer_enc, tokenizer_dec, max_length_kor, max_length_eng)

    # evaluate
    bleu = evaluate.load("bleu")

    results = bleu.compute(predictions=predictions, references=decoder_target_test)

    # save model
