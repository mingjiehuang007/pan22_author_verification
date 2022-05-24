import random
import numpy as np
from bert4keras.backend import keras, set_gelu, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, AdaFactorV1
from bert4keras.snippets import sequence_padding, DataGenerator
from tensorflow.keras.layers import Dropout, Dense, Lambda, LSTM, GlobalAveragePooling1D
from bert4keras.snippets import  text_segmentate
from tqdm import tqdm
import os
import json
import tensorflow as tf
from collection import load_text_data, filter_info_pairs, filter_info_truth, count_angular_symbol,count_special_symbol


# gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
# tf.compat.v1.disable_eager_execution()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_an_sp(path_of_pairs, path_of_truth):
    pairs = load_text_data(path_of_pairs)
    truth = load_text_data(path_of_truth)
    dt_list, max_len, min_len, sen_list = filter_info_pairs(pairs)
    author_count_list, author_count, label_count = filter_info_truth(truth)
    angular_symbol_list = count_angular_symbol(sen_list)  # 尖括号符号及数量
    special_symbol_list = count_special_symbol(sen_list)  # 特殊符号及数量
    angular_list = []  # 所有尖括号符号的列表
    for i in angular_symbol_list:
        angular_list.append(i[0])
    special_list = []  # 所有特殊符号的列表
    for i in special_symbol_list:
        special_list.append(i[0])
    return angular_list, special_list


def eliminate_angular(sen, angular_list, special_list):
    for an in angular_list:
        if an in sen:
            sen = sen.replace(an, '')
    for sp in special_list:
        if sp in sen:
            sen = sen.replace(sp, '')
    return sen


def get_data(truth_path, text_path):
    angl_list, spcl_list = get_an_sp(text_path, truth_path)

    truth=[]
    with open(truth_path,'r',encoding='utf-8') as f:
        for l in f:
            data = json.loads(l)
            truth.append((data['id'], data['same'], data['authors']))

    index=0

    with open(text_path,'r', encoding='utf-8') as f:
        datas=[]
        for l in tqdm(f):
            data = json.loads(l)
            if truth[index][0]==data['id']:
                pair_1 = eliminate_angular(data['pair'][0], angl_list, spcl_list)
                pair_2 = eliminate_angular(data['pair'][1], angl_list, spcl_list)

                text1 = text_segmentate(pair_1, maxlen=510, seps='.?!')
                text2 = text_segmentate(pair_2, maxlen=510, seps='.?!')

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

                datas.append((text1, text2, int(truth[index][-2]), str(data['id']), truth[index][-2], truth[index][-1]))

            index+=1

    return datas

def random_(text):
    np.random.seed(1)
    np.random.shuffle(text)

    return text
truth_path = r'D:\NLP\数据集\PAN2022数据集\pan22-authorship-verification-training-dataset\truth.jsonl'
text_path = r'D:\NLP\数据集\PAN2022数据集\pan22-authorship-verification-training-dataset\pairs.jsonl'
data = random_(get_data(truth_path,text_path))
data = data[:8585]    # 这里是取出70%当训练集进行微调
angl_list, spcl_list = get_an_sp(text_path, truth_path)


maxlen = 64     # 原256
batch_size = 32  # 原64


config_path = r'D:\NLP\models\BERT\cased_L-12_H-768_A-12\bert_config.json'
checkpoint_path = r'D:\NLP\models\BERT\cased_L-12_H-768_A-12\bert_model.ckpt'
dict_path = r'D:\NLP\models\BERT\cased_L-12_H-768_A-12\vocab.txt'

 # 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=False)

def generator():
    class data_generator(DataGenerator):
        """数据生成器
        """
        def __iter__(self, random=False):
            batch_token_ids, batch_segment_ids, batch_labels = [], [], []
            for is_end, (text1, text2, label, id, _, _) in self.sample(random):
                for index, sent in enumerate(text1[:30]):
                    token_ids, segment_ids = tokenizer.encode(text1[index], text2[index], maxlen=maxlen)
                    batch_token_ids.append(token_ids)
                    batch_segment_ids.append(segment_ids)
                    batch_labels.append([label])
                    if len(batch_labels) == self.batch_size or is_end:
                        batch_token_ids = sequence_padding(batch_token_ids)
                        batch_segment_ids = sequence_padding(batch_segment_ids)
                        batch_labels = sequence_padding(batch_labels)

                        yield [batch_token_ids, batch_segment_ids], batch_labels
                        batch_token_ids, batch_segment_ids, batch_labels = [], [], []

    generator = data_generator(data, batch_size)

    class Evaluator(keras.callbacks.Callback):
        """评估与保存
        """
        def __init__(self):
            self.best_val_acc = 0.

        def on_epoch_end(self, epoch, logs=None):

            model.save_weights(str(epoch) +'_model.weights')

    evaluator = Evaluator()

    bert0 = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        return_keras_model=False,
        num_hidden_layers=12,
    )

    output = Lambda(lambda x: x[:, 0])(bert0.model.output)
    output = Dropout(rate=0.4)(output)                      # 原0.4
    output = Dense(units=2, activation='softmax', )(output)

    model = keras.models.Model(bert0.model.inputs, output)

    model.compile(
        optimizer=Adam(2e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    model.fit(
        generator.forfit(),
        steps_per_epoch=len(generator),
        epochs=5,
        callbacks=[evaluator]
    )


if __name__ == '__main__':
    generator()
    print('')
