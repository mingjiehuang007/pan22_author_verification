import random
import numpy as np
from bert4keras.backend import keras, set_gelu, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, AdaFactorV1
from bert4keras.snippets import sequence_padding, DataGenerator
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Dropout, Dense, Lambda, LSTM, GlobalAveragePooling1D
from bert4keras.snippets import  text_segmentate
from tqdm import tqdm
import os
import json
from collection import load_text_data, filter_info_pairs, filter_info_truth, count_angular_symbol,count_special_symbol


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_an_sp(path_of_pairs):
    pairs = load_text_data(path_of_pairs)
    dt_list, max_len, min_len, sen_list = filter_info_pairs(pairs)
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


def dt_num(discourse_type):
    dt_list = ["essay", "email", "text_message", "memo"]
    return dt_list.index(discourse_type)


def get_data(truth_path, text_path, dt_type):
    angl_list, spcl_list = get_an_sp(text_path)

    truth = []
    with open(truth_path, 'r', encoding='utf-8') as f:
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

                dt_type_1 = data['discourse_types'][0]
                dt_type_2 = data['discourse_types'][1]

                text1 = text_segmentate(pair_1, maxlen=510, seps='.?!')     # 按字符串长度切片
                text2 = text_segmentate(pair_2, maxlen=510, seps='.?!')

                if dt_type_1 == 'essay':
                    min_len = 20
                else:
                    min_len = 3

                while len(text1) < min_len or len(text2) < min_len:
                    if len(text1) < min_len:
                        n_text1 = []
                        for i in range(min_len):
                            for sent in text1:
                                n_text1.append(sent)
                        text1 = n_text1
                    elif len(text2) < min_len:
                        n_text2 = []
                        for i in range(min_len):
                            for sent in text2:
                                n_text2.append(sent)
                        text2 = n_text2

                datas.append((text1, text2, int(truth[index][-2]), str(data['id']), dt_type_1, dt_type_2))

            index+=1

    datas_elected = []
    if dt_type == 'long':
        for i in datas:
            if i[4] == 'essay':
                datas_elected.append(i)
    elif dt_type == 'short':
        for i in datas:
            if i[4] != 'essay':
                datas_elected.append(i)

    return datas_elected

def random_(text):
    np.random.seed(1)
    np.random.shuffle(text)

    return text


def build_bert_model_2(config_path,checkpoint_path):
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
    return model



maxlen = 64     # 原256
batch_size = 32  # 原64


config_path = r'D:\NLP\models\BERT\cased_L-12_H-768_A-12\bert_config.json'
checkpoint_path = r'D:\NLP\models\BERT\cased_L-12_H-768_A-12\bert_model.ckpt'
dict_path = r'D:\NLP\models\BERT\cased_L-12_H-768_A-12\vocab.txt'

 # 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=False)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label, id, _, _) in self.sample(random):
            for index, sent in enumerate(text1[:20]):
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


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            model.save_weights(str(epoch) +'_model.weights')
        # model.save_weights('best_model.weights')


if __name__ == '__main__':
    # 加载数据集
    text_path = r'D:\NLP\数据集\PAN2022数据集\process_data\new_train.jsonl'
    truth_path = r'D:\NLP\数据集\PAN2022数据集\process_data\new_train_truth.jsonl'

    data = random_(get_data(truth_path, text_path, 'long'))
    train_data_len = int(len(data) * 0.85)
    # data = data[:train_data_len]    # 这里是取出70%当训练集进行微调

    # generator = data_generator(data, batch_size)
    train_generator = data_generator(data[:train_data_len], batch_size)
    valid_generator = data_generator(data[train_data_len:], batch_size)

    evaluator = Evaluator()

    model = build_bert_model_2(config_path, checkpoint_path)

    model.compile(
        optimizer=Adam(2e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    earlystop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=2,
        mode='min'
    )
    bast_model_filepath = 'checkpoint2/best_model.weights'
    checkpoint = keras.callbacks.ModelCheckpoint(
        bast_model_filepath,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='min'
    )

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=20,
        validation_data=valid_generator.forfit(),
        validation_steps=len(valid_generator),
        shuffle=True,
        callbacks=[checkpoint, earlystop]
    )
