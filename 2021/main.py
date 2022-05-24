#需要命令行运行 python -c test_data_path -o output_answers_path

import random
import numpy as np
from bert4keras.backend import keras, set_gelu, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, AdaFactorV1
from bert4keras.snippets import sequence_padding, DataGenerator, text_segmentate, to_array
from keras.layers import Dropout, Dense, Lambda, LSTM, GlobalAveragePooling1D
import tensorflow as tf
import argparse
import gc
from bert4keras.snippets import text_segmentate
from tqdm import tqdm
import os
from keras.models import load_model, Model

import json


def get_data(test_path):

    with open(test_path, 'r') as f:

        datas = []
        for l in f:
            data = json.loads(l)
            text1 = text_segmentate(data['pair'][0], maxlen=510, seps='.?!')
            text2 = text_segmentate(data['pair'][1], maxlen=510, seps='.?!')
            while len(text1) < 30 or len(text2) < 30:
                if len(text1) < 30:
                    n_text1 = []
                    for i in range(30):
                        for sent in text1:
                            n_text1.append(sent)
                    text1 = n_text1
                elif len(text2) < 30:
                    n_text2 = []
                    for i in range(30):
                        for sent in text2:
                            n_text2.append(sent)
                    text2 = n_text2
            datas.append((text1, text2, str(data['id'])))

        return datas


def main():

    parser = argparse.ArgumentParser(description='Evaluation script AA@PAN2021')
    parser.add_argument('-c', type=str,
                        help='Path to the jsonl-file with pairs')
    parser.add_argument('-o', type=str,
                        help='Path to the dir with output')
    args = parser.parse_args()

    test_path = args.c
    answers_path = args.o

    test_data = get_data(test_path + '/pairs.jsonl')
    print(len(test_data))

    maxlen = 256
    batch_size = 1

    config_path = '../cased_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = '../cased_L-12_H-768_A-12/bert_model.ckpt'
    dict_path = '../cased_L-12_H-768_A-12/vocab.txt'

    tokenizer = Tokenizer(dict_path, do_lower_case=False)

    class data_generator(DataGenerator):

        def __iter__(self, random=False):
            batch_token_ids, batch_segment_ids = [], []
            for is_end, (text1, text2, _) in self.sample(random):
                for index, sent in enumerate(text1[:30]):
                    token_ids, segment_ids = tokenizer.encode(text1[index], text2[index], maxlen=maxlen)
                    batch_token_ids.append(token_ids)
                    batch_segment_ids.append(segment_ids)
                    if len(batch_token_ids)//30 == self.batch_size:
                        batch_token_ids = sequence_padding(batch_token_ids)
                        batch_segment_ids = sequence_padding(batch_segment_ids)

                        yield [batch_token_ids, batch_segment_ids]
                        batch_token_ids, batch_segment_ids = [], []

    # 转换数据集
    test_generator = data_generator(test_data, batch_size)

    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        return_keras_model=False,
        num_hidden_layers=12,
    )

    output = Lambda(lambda x: x[:, 0])(bert.model.output)
    output = Dropout(rate=0.2)(output)
    output = Dense(units=2, activation='softmax', )(output)

    model = keras.models.Model(bert.model.inputs, output)

    model.load_weights('/home/peng21/home/4_model.weights')  #加载微调后的权重

    cls_layer = Model(inputs=model.input, outputs=model.layers[-3].output)

    n_model = load_model('/home/peng21/home/final_model.h5')

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(NpEncoder, self).default(obj)

    def get_sents_represent(generator, texts_num):
        sents_r = np.array([])
        print('the progress of generating text representation: ')
        for x_true in tqdm(generator):

            sent_r = cls_layer.predict(x_true)
            sents_r = np.append(sents_r, sent_r)

        sents_rs = sents_r.reshape(texts_num, 30, 768)

        return sents_rs

    test_sents_r = get_sents_represent(test_generator, len(test_data))

    res = n_model.predict(test_sents_r)
    n_res = np.argmax(res, axis=-1)

    with open(answers_path + '/answers.jsonl', 'w') as f:
        for index, value in enumerate(test_data):
            dic = {"id": value[-1], "value": n_res[index]}
            f.write(json.dumps(dic, cls=NpEncoder))
            f.write('\n')


if __name__ == '__main__':
    main()
