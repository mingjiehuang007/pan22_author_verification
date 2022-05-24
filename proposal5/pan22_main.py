#需要命令行运行 python -i test_data_path -o output_answers_path

import numpy as np
from bert4keras.backend import keras, set_gelu, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import sequence_padding, DataGenerator, text_segmentate, to_array
from tensorflow.keras.layers import Dropout, Dense, Lambda, LSTM, GlobalAveragePooling1D
import argparse
from bert4keras.snippets import text_segmentate
from tqdm import tqdm
from tensorflow.keras.models import load_model, Model
import json, re
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


def filter_info_truth(list_src):
    label_count = [0, 0]
    author_list = []
    count_list = []
    for line in list_src:
        data1 = json.loads(line)
        if data1["same"]:
            label_count[0] += 1                     # 正例数量
        else:
            label_count[1] += 1                     # 反例数量
        for i in range(2):
            author = data1["authors"][i]
            if len(author_list) == 0:
                author_list.append(author)
                count_list.append(1)
            elif author in author_list:
                dis_pos = author_list.index(author)
                count_list[dis_pos] += 1
            else:
                author_list.append(author)
                count_list.append(1)
    res = []                                         # 作者列表
    for i in range(len(author_list)):
        res.append([author_list[i], count_list[i]])  # 各作者写作数量

    return res, len(res), label_count


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


def get_ids(test_path):
    with open(test_path, 'r', encoding='utf-8') as f:
        ids = []
        for l in f:
            data = json.loads(l)
            ids.append(str(data['id']))
    return ids


def build_bert_model_1(config_path,checkpoint_path):
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        return_keras_model=False,
        num_hidden_layers=12,
    )

    # output = Lambda(lambda x: x[:, 0])(bert.model.output)
    output = Lambda(lambda x: x[:, 1:-1],)(bert.model.output)
    output = GlobalAveragePooling1D()(output)
    output = Dense(units=768, activation='relu')(output)
    output = Dense(units=2, activation='softmax')(output)
    model = keras.models.Model(bert.model.inputs, output)
    return model


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

            datas.append((text1, text2, str(data['id']), dt_type_1))

    return datas


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
    test_data = test_data[:10]

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
                for index_1, sent_1 in enumerate(text1):
                    for index_2, sent_2 in enumerate(text2):
                        token_ids, segment_ids = tokenizer.encode(sent_1, sent_2, maxlen=maxlen)
                        batch_token_ids.append(token_ids)
                        batch_segment_ids.append(segment_ids)
                        if len(batch_token_ids) == self.batch_size:
                            batch_token_ids = sequence_padding(batch_token_ids)
                            batch_segment_ids = sequence_padding(batch_segment_ids)

                            yield [batch_token_ids, batch_segment_ids]
                            batch_token_ids, batch_segment_ids = [], []

    # 转换数据集
    test_generator = data_generator(test_data, batch_size)

    model = build_bert_model_1(config_path, checkpoint_path,)
    model.load_weights('checkpoint1/best_model.weights')
    # model.load_weights('/home/peng21/home/4_model.weights')  #加载微调后的权重

    cls_layer = Model(inputs=model.input, outputs=model.layers[-1].output)
    # cls_layer = Model(inputs=model.input, outputs=model.layers[-4].output)

    predict_model = load_model('final_model.h5')
    # predict_model = load_model('/home/peng21/home/final_model.h5')

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

    def count_pairs(datum):
        pair_count = []
        for i in datum:
            pair_count.append(len(i[0]) * len(i[1]))

        return pair_count

    def get_sents_similarity(generator, texts_num):
        sents_r = np.array([])
        for x_true in tqdm(generator):
            sent_r = cls_layer.predict(x_true)
            sents_r = np.append(sents_r, sent_r)

        res = sents_r.reshape(texts_num, 2)

        return res

    def combine_prediction(predict_res, texts_num, pair_count):
        res = np.array([])
        start = 0
        for c in pair_count:
            end = start + c
            ans = predict_res[start:end]
            count_0 = np.sum(ans == 0)
            count_1 = np.sum(ans == 1)
            count_01 = np.array([count_0, count_1])
            count_01 = np.exp(count_01) / sum(np.exp(count_01))  # softmax
            answer_arr = np.append(count_01, c)
            res = np.append(res, answer_arr)
            start = end

        res = res.reshape(texts_num, 3)

        return res


    test_pair_count = count_pairs(test_data)
    test_pair_num = np.sum(test_pair_count)
    test_sents_s = get_sents_similarity(test_generator, test_pair_num)
    test_sents_ans = combine_prediction(test_sents_s, len(test_data), test_pair_count)

    res = predict_model.predict(test_sents_ans)
    n_res = np.argmax(res, axis=-1)

    # ids = get_ids(test_path + '/pairs.jsonl')
    ids = get_ids(r'D:\NLP\数据集\PAN2022数据集\process_data\pairs.jsonl')

    # with open(answers_path + '/answers.jsonl', 'w', encoding='utf-8') as f:
    with open('answers.jsonl', 'w', encoding='utf-8') as f:
        for index, value in enumerate(test_data):
            dic = {"id": value[2], "value": n_res[index]}
            f.write(json.dumps(dic, cls=NpEncoder))
            f.write('\n')


if __name__ == '__main__':
    main()
