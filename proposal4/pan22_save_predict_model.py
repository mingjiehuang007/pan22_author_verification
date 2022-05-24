import numpy as np
import json, re
from bert4keras.snippets import text_segmentate
from bert4keras.backend import keras, set_gelu, K
from tensorflow.keras.layers import LSTM, Bidirectional
from tqdm import tqdm
from tensorflow.keras.layers import Dense,GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession

# 此3句代码用于解决NotFoundError: No algorithm worked! when using Conv2D的报错
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


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


def eliminate_short_sens(sen_list, min_len):
    sen_len_list = []
    for i in sen_list:
        sen_len_list.append(len(i))

    if max(sen_len_list) >= min_len:
        new_sen_list = []
        for i in sen_list:
            if len(i) >= min_len:
                new_sen_list.append(i)
        return new_sen_list
    else:
        return sen_list


def get_data(truth_path, text_path):
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

                datas.append((text1, text2, int(truth[index][-2]), str(data['id']), dt_type_1, dt_type_2))

            index+=1

    return datas


text_path = r'D:\NLP\数据集\PAN2022数据集\process_data\new_traindata_text.jsonl'
truth_path = r'D:\NLP\数据集\PAN2022数据集\process_data\new_traindata_truth.jsonl'
data = get_data(truth_path, text_path)
train_data_len = int(len(data) * 0.85)


def count_pairs(datum):
    pair_count = []
    for i in datum:
        pair_count.append(len(i[0]) * len(i[1]))

    return pair_count


def get_labels(data, pair_count):
    labels = []
    label_in_text = []
    for i in data:
        label_in_text.append(i[2])
    for index, c in enumerate(pair_count):
        for j in range(c):
            labels.append(label_in_text[index])

    labels = np.array(labels)
    return labels


train_pair_count = count_pairs(data[:train_data_len])
valid_pair_count = count_pairs(data[train_data_len:])
train_sents_r = np.load("checkpoint1/train_sents_rs.npy",mmap_mode = 'r')
train_labels=get_labels(data[:train_data_len], train_pair_count)
valid_sents_r = np.load("checkpoint1/test_sents_rs.npy",mmap_mode = 'r')
valid_labels=get_labels(data[train_data_len:], valid_pair_count)


callbacks = [
    ModelCheckpoint(
        f'weights1.h5',
        monitor='val_sparse_categorical_accuracy',
        save_weights_only=True,
        save_best_only=True,
        verbose=1,
        mode='max'),
]

earlystop = keras.callbacks.EarlyStopping(
    monitor='val_sparse_categorical_accuracy',
    patience=20,
    verbose=2,
    mode='max'
)


model = Sequential()
model.add(GlobalAveragePooling1D())
model.add(Dense(units=16, activation='relu', ))
model.add(Dense(units=2, activation='softmax', ))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
model.fit(train_sents_r,train_labels,epochs=500,validation_data=(valid_sents_r,valid_labels),callbacks=[callbacks, earlystop])

model.save('predict_model.h5')
