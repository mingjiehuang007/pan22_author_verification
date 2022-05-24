import json
import re

from bert4keras.backend import keras, set_gelu
from tensorflow.keras.models import  Model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator, text_segmentate, to_array
from tensorflow.keras.layers import Dropout, Dense, Lambda, LSTM, GlobalAveragePooling1D
import numpy as np
from tqdm import tqdm
from bert4keras.models import build_transformer_model
import tensorflow.python.util.deprecation as deprecation
from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession

# 此3句代码用于解决NotFoundError: No algorithm worked! when using Conv2D的报错
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

deprecation._PRINT_DEPRECATION_WARNINGS = False


set_gelu('tanh')

def textcnn(inputs,kernel_initializer):
    # 3,4,5
    cnn1 = keras.layers.Conv1D(
            256,				# fileter数量
            3,					# 卷积核大小
            strides=1,			# 步幅
            padding='same',		# 方法
            activation='relu',
            kernel_initializer=kernel_initializer
        )(inputs) # shape=[batch_size,maxlen-2,256]
    cnn1 = keras.layers.GlobalMaxPooling1D()(cnn1)  # shape=[batch_size,256]

    cnn2 = keras.layers.Conv1D(
            256,
            4,
            strides=1,
            padding='same',
            activation='relu',
            kernel_initializer=kernel_initializer
        )(inputs)
    cnn2 = keras.layers.GlobalMaxPooling1D()(cnn2)

    cnn3 = keras.layers.Conv1D(
            256,
            5,
            strides=1,
            padding='same',
            kernel_initializer=kernel_initializer
        )(inputs)
    cnn3 = keras.layers.GlobalMaxPooling1D()(cnn3)

    output = keras.layers.concatenate(
        [cnn1,cnn2,cnn3],
        axis=-1)
    output = keras.layers.Dropout(0.2)(output)
    return output


def build_bert_model_3(config_path,checkpoint_path,class_nums):
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model='electra',
        return_keras_model=False)

    cls_features = keras.layers.Lambda(
        lambda x:x[:,0],
        name='cls-token'
        )(bert.model.output) #shape=[batch_size,768]
    all_token_embedding = keras.layers.Lambda(
        lambda x:x[:,1:-1],
        name='all-token'
        )(bert.model.output) #shape=[batch_size,maxlen-2,768]

    cnn_features = textcnn(
        all_token_embedding,bert.initializer) #shape=[batch_size,cnn_output_dim]
    concat_features = keras.layers.concatenate(
        [cls_features,cnn_features],
        axis=-1)

    dense = keras.layers.Dense(
            units=256,
            activation='relu',
            kernel_initializer=bert.initializer
        )(concat_features)

    output = keras.layers.Dense(
            units=class_nums,
            activation='softmax',
            kernel_initializer=bert.initializer
        )(dense)

    model = keras.models.Model(bert.model.input,output)

    return model


def load_text_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        s = []
        for l in f:
            s.append(l.strip())
        return s


def filter_info_pairs(list_src):
    discourse_list = []
    count_list = []
    sen_list = []
    sen_len_list = []
    for line in list_src:
        data1 = json.loads(line)
        for i in range(2):
            discourse = data1["discourse_types"][i]
            sen_list.append(data1["pair"][i])
            sen_len_list.append(len(data1["pair"][i]))
            if len(discourse_list) == 0:
                discourse_list.append(discourse)
                count_list.append(1)
            elif discourse in discourse_list:
                dis_pos = discourse_list.index(discourse)
                count_list[dis_pos] += 1
            else:
                discourse_list.append(discourse)
                count_list.append(1)

    max_sen_len = max(sen_len_list)             # 最大文本长度
    min_sen_len = min(sen_len_list)             # 最小文本长度
    res = []                                    # 记录话语类别以及数量
    for i in range(len(discourse_list)):
        res.append([discourse_list[i], count_list[i]])
    return res, max_sen_len, min_sen_len, sen_list


def count_angular_symbol(str_list):     # 统计带<>符号的特殊文本的数量
    symbol_list = []
    symbol_count = []
    for str_src in str_list:
        flag = False
        sym = ''
        for c in str_src:
            if c == "<":
                flag = True
            if flag:
                sym += c
            if c == ">":
                flag = False
                if sym not in symbol_list:
                    symbol_list.append(sym)
                    symbol_count.append(1)
                elif sym in symbol_list:
                    sym_pos = symbol_list.index(sym)
                    symbol_count[sym_pos] += 1
                sym = ''
    symbol_list_count = []
    for i in range(len(symbol_list)):
        symbol_list_count.append([symbol_list[i], symbol_count[i]])
    return symbol_list_count


def count_special_symbol(str_list):               # 统计其他的特殊文本的数量
    special_list = []
    special_count = []
    for str_src in str_list:
        str_src = re.sub('[a-zA-Z0-9’!"#$%&\'()*+,\-–./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+', "", str_src)
        for c in str_src:
            if c not in special_list:
                special_list.append(c)
                special_count.append(1)
            elif c in special_list:
                special_pos = special_list.index(c)
                special_count[special_pos] += 1
    special_list_count = []
    for i in range(len(special_list)):
        special_list_count.append([special_list[i], special_count[i]])
    return special_list_count


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


min_len = 4


def get_data(truth_path, text_path):
    angl_list, spcl_list = get_an_sp(text_path)

    truth = []
    with open(truth_path, 'r', encoding='utf-8') as f:
        for l in f:
            data = json.loads(l)
            truth.append((data['id'], data['same'], data['authors']))

    index = 0

    with open(text_path, 'r', encoding='utf-8') as f:
        datas = []
        for l in tqdm(f):
            data = json.loads(l)
            if truth[index][0] == data['id']:
                pair_1 = eliminate_angular(data['pair'][0], angl_list, spcl_list)
                pair_2 = eliminate_angular(data['pair'][1], angl_list, spcl_list)

                dt_type_1 = data['discourse_types'][0]
                dt_type_2 = data['discourse_types'][1]

                text1 = text_segmentate(pair_1, maxlen=510, seps='.?!')  # 按字符串长度切片
                text2 = text_segmentate(pair_2, maxlen=510, seps='.?!')

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

                text1_op_ed = text1[0:2]
                text1_op_ed.append(text1[-2])
                text1_op_ed.append(text1[-1])
                text2_op_ed = text2[0:2]
                text2_op_ed.append(text2[-2])
                text2_op_ed.append(text2[-1])

                datas.append((text1_op_ed, text2_op_ed, int(truth[index][-2]), str(data['id']), dt_type_1, dt_type_2))

            index += 1

    return datas



maxlen = 256
batch_size = 30

config_path = r'D:\NLP\models\BERT\electra_small\electra_config.json'
checkpoint_path = r'D:\NLP\models\BERT\electra_small\electra_small'
dict_path = r'D:\NLP\models\BERT\electra_small\vocab.txt'

 # 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label, id, _, _) in self.sample(random):
            for index, sent in enumerate(text1):
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


text_path = r'D:\NLP\数据集\PAN2022数据集\process_data\new_train.jsonl'
truth_path = r'D:\NLP\数据集\PAN2022数据集\process_data\new_train_truth.jsonl'
data = get_data(truth_path, text_path)
train_data_len = int(len(data) * 0.85)
train_generator = data_generator(data[:train_data_len], batch_size)
test_generator = data_generator(data[train_data_len:], batch_size)


model = build_bert_model_3(config_path, checkpoint_path, class_nums=2)
model.load_weights('checkpoint3/best_model.weights')  #加载微调后的权重

cls_layer = Model(inputs=model.input, outputs=model.layers[-2].output)  #后面会从cls_layer提取cls向量作为句子表示

def get_sents_represent(generator, texts_num):
    sents_r = np.array([])
    for x_true, y_true in tqdm(generator):
        sent_r = cls_layer.predict(x_true)
        sents_r = np.append(sents_r, sent_r)
    sents_rs = sents_r.reshape(texts_num, 4, 256)

    return sents_rs


train_sents_rs = get_sents_represent(train_generator, len(data[:train_data_len]))
test_sents_rs = get_sents_represent(test_generator, len(data[train_data_len:]))

np.save("checkpoint3/train_sents_rs.npy",train_sents_rs)
np.save("checkpoint3/test_sents_rs.npy",test_sents_rs)

