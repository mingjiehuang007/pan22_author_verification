#需要命令行运行 python -i test_data_path -o output_answers_path

import numpy as np
from bert4keras.backend import keras, set_gelu, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import sequence_padding, DataGenerator, text_segmentate, to_array
from keras.layers import Dropout, Dense, Lambda, LSTM, GlobalAveragePooling1D
import argparse
from bert4keras.snippets import text_segmentate
from tqdm import tqdm
from keras.models import load_model, Model
import json
from collection import load_text_data, filter_info_pairs, filter_info_truth, count_angular_symbol,count_special_symbol

from pan22_model1_bert_model import build_bert_model
from pan22_model2_bert_model import build_bert_model_2

from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession

# 此3句代码用于解决NotFoundError: No algorithm worked! when using Conv2D的报错
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

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


def get_ids(test_path):
    with open(test_path, 'r', encoding='utf-8') as f:
        ids = []
        for l in f:
            data = json.loads(l)
            ids.append(str(data['id']))
    return ids


def get_test_data(test_path):
    angl_list, spcl_list = get_an_sp(test_path)

    with open(test_path, 'r', encoding='utf-8') as f:

        datas = []
        for l in f:
            data = json.loads(l)
            pair_1 = eliminate_angular(data['pair'][0], angl_list, spcl_list)
            pair_2 = eliminate_angular(data['pair'][1], angl_list, spcl_list)

            dt_type_1 = data['discourse_types'][0]
            dt_type_2 = data['discourse_types'][1]

            text1 = text_segmentate(pair_1, maxlen=510, seps='.?!')  # 按字符串长度切片
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

            datas.append((text1, text2, str(data['id']), dt_type_1))

    return datas


def split_data(datas):
    long_text_data, short_text_data = [], []
    for data in datas:
        if data[3] != 'essay':
            short_text_data.append(data)
        else:
            long_text_data.append(data)
    return short_text_data, long_text_data


def main():

    parser = argparse.ArgumentParser(description='Evaluation script AA@PAN2021')
    parser.add_argument('-i', type=str,
                        help='Path to the jsonl-file with pairs')
    parser.add_argument('-o', type=str,
                        help='Path to the dir with output')
    args = parser.parse_args()

    test_path = args.i
    answers_path = args.o

    # test_data = get_test_data(test_path + '/pairs.jsonl')
    test_data = get_test_data(r'D:\NLP\数据集\PAN2022数据集\process_data\pairs.jsonl')
    print(len(test_data))

    maxlen = 256
    batch_size = 1

    config_path = r'D:\NLP\models\BERT\cased_L-12_H-768_A-12\bert_config.json'
    checkpoint_path = r'D:\NLP\models\BERT\cased_L-12_H-768_A-12\bert_model.ckpt'
    dict_path = r'D:\NLP\models\BERT\cased_L-12_H-768_A-12\vocab.txt'

    tokenizer = Tokenizer(dict_path, do_lower_case=False)

    class data_generator(DataGenerator):

        def __iter__(self, random=False):
            batch_token_ids, batch_segment_ids = [], []
            for is_end, (text1, text2, _, dt_type_1) in self.sample(random):
                if dt_type_1 == 'essay':
                    min_len = 20
                else:
                    min_len = 3
                for index, sent in enumerate(text1[:min_len]):
                    token_ids, segment_ids = tokenizer.encode(text1[index], text2[index], maxlen=maxlen)
                    batch_token_ids.append(token_ids)
                    batch_segment_ids.append(segment_ids)
                    if len(batch_token_ids) == self.batch_size:
                        batch_token_ids = sequence_padding(batch_token_ids)
                        batch_segment_ids = sequence_padding(batch_segment_ids)

                        yield [batch_token_ids, batch_segment_ids]
                        batch_token_ids, batch_segment_ids = [], []

    # 转换数据集
    short_test_data, long_test_data = split_data(test_data)
    short_test_generator = data_generator(short_test_data, batch_size)
    long_test_generator = data_generator(long_test_data, batch_size)

    model_1 = build_bert_model(config_path, checkpoint_path, class_nums=2)
    model_2 = build_bert_model_2(config_path, checkpoint_path)

    model_1.load_weights('checkpoint1/best_model.weights')
    model_2.load_weights('checkpoint2/best_model.weights')
    # model.load_weights('/home/peng21/home/4_model.weights')  #加载微调后的权重

    cls_layer_1 = Model(inputs=model_1.input, outputs=model_1.layers[-2].output)
    cls_layer_2 = Model(inputs=model_2.input, outputs=model_2.layers[-3].output)
    # cls_layer = Model(inputs=model.input, outputs=model.layers[-3].output)

    n_model = load_model('final_model.h5')
    # n_model = load_model('/home/peng21/home/final_model.h5')

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

    def get_sents_represent(generator, texts_num, text_type):
        sents_r = np.array([])
        print('the progress of generating text representation: ')
        if text_type == 'short':
            cls_layer = cls_layer_1
            min_len = 3
        else:
            cls_layer = cls_layer_2
            min_len = 20

        for x_true in tqdm(generator):
            sent_r = cls_layer.predict(x_true)
            sents_r = np.append(sents_r, sent_r)

        sents_rs = sents_r.reshape(texts_num, min_len, 768)
        sents_rs = np.mean(sents_rs, axis=1, keepdims=True)
        return sents_rs


    test_sents_r_1 = get_sents_represent(short_test_generator, len(short_test_data), 'short')
    test_sents_r_2 = get_sents_represent(long_test_generator, len(long_test_data), 'long')

    res_1 = n_model.predict(test_sents_r_1)
    n_res_1 = np.argmax(res_1, axis=-1)

    res_2 = n_model.predict(test_sents_r_2)
    n_res_2 = np.argmax(res_2, axis=-1)

    # ids = get_ids(test_path + '/pairs.jsonl')
    ids = get_ids(r'D:\NLP\数据集\PAN2022数据集\process_data\pairs.jsonl')

    ids_1, ids_2 = [], []
    for i in short_test_data:
        ids_1.append(i[2])
    for i in long_test_data:
        ids_2.append(i[2])


    # 合并两个分类器的结果
    res_list = []
    for i in ids:
        flag = False
        for index, id_1 in enumerate(ids_1):
            if id_1 == i:
                res_list.append(n_res_1[index])
                flag = True
                break

        if flag == False:
            for index, id_2 in enumerate(ids_2):
                if id_2 == i:
                    res_list.append(n_res_2[index])
                    break


    # with open(answers_path + '/answers.jsonl', 'w', encoding='utf-8') as f:
    with open('answers.jsonl', 'w', encoding='utf-8') as f:
        for index, value in enumerate(test_data):
            dic = {"id": value[2], "value": res_list[index]}
            f.write(json.dumps(dic, cls=NpEncoder))
            f.write('\n')


if __name__ == '__main__':
    main()
