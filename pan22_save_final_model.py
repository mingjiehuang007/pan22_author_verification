from bert4keras.snippets import sequence_padding, DataGenerator, text_segmentate, to_array
from keras.layers import Dropout, Dense, Lambda, LSTM, GlobalAveragePooling1D
import numpy as np
import json, re
from tqdm import tqdm
from bert4keras.models import build_transformer_model
import tensorflow.python.util.deprecation as deprecation
from keras.callbacks import ModelCheckpoint
from keras import Sequential


deprecation._PRINT_DEPRECATION_WARNINGS = False


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


def get_segments(datas):
    new_datas = []
    for data in datas:
        text1 = data[0]
        text2 = data[1]
        for i in text1:
            for j in text2:
                new_datas.append((i,j,data[2],data[3],data[4],data[5]))

    return new_datas


text_path = r'D:\NLP\数据集\PAN2022数据集\process_data\new_traindata_text.jsonl'
truth_path = r'D:\NLP\数据集\PAN2022数据集\process_data\new_traindata_truth.jsonl'
data_origin = get_data(truth_path, text_path)

train_data_len = int(len(data_origin) * 0.7)

labels=[]
for i in data_origin:
    labels.append(i[2])
train_labels=np.array(labels[:train_data_len])
valid_labels=np.array(labels[train_data_len:])

train_sents_s = np.load("checkpoint1/train_sents_s.npy",mmap_mode = 'r')
valid_sents_s = np.load("checkpoint1/valid_sents_s.npy",mmap_mode = 'r')

callbacks = [
    ModelCheckpoint(
        f'checkpoint1/weights1.h5',
        monitor='val_sparse_categorical_accuracy',
        save_weights_only=True,
        save_best_only=True,
        verbose=1,
        mode='max'),
]

model = Sequential()
model.add(GlobalAveragePooling1D())
model.add(Dense(units=16, activation='relu', ))
model.add(Dense(units=2, activation='softmax', ))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
model.fit(train_sents_s,train_labels,epochs=500,validation_data=(valid_sents_s,valid_labels),callbacks=callbacks)

model.save('checkpoint1/final_model.h5')
