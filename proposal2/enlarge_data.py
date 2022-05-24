import json, os, re
import pandas as pd
from tqdm import tqdm


def load_text_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        s = []
        for l in f:
            s.append(l.strip())
        return s


def get_origin_data(truth_path, text_path):
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

                text_1 = data['pair'][0]
                text_2 = data['pair'][1]

                dt_type_1 = data['discourse_types'][0]
                dt_type_2 = data['discourse_types'][1]

                author_1 = truth[index][2][0]
                author_2 = truth[index][2][1]


                datas.append((text_1, text_2, dt_type_1, dt_type_2, author_1, author_2))

            index+=1

    return datas


def create_author_file(datas, author_path):
    with open(author_path, 'w', encoding='utf-8') as f:
        writer_temp = []
        for data in tqdm(datas):
            text_1 = data[0]
            text_2 = data[1]
            dt_type_1 = data[2]
            dt_type_2 = data[3]
            author_1 = data[4]
            author_2 = data[5]
            writer_temp.append((author_1, dt_type_1, text_1))
            writer_temp.append((author_2, dt_type_2, text_2))

        writer_temp = list(set(writer_temp))
        for i in writer_temp:
            dic = {'author': i[0], 'discourse_type': i[1], 'text': i[2]}
            f.write(json.dumps(dic) + '\n')


def create_new_data(new_truth_path, new_pairs_path):
    f_truth = open(new_truth_path, 'w', encoding='utf-8')
    f_pairs = open(new_pairs_path, 'w', encoding='utf-8')

    with open(r'D:\NLP\数据集\PAN2022数据集\process_data\author_dt_sen.jsonl', 'r', encoding='utf-8') as f:
        data = []
        for l in tqdm(f):
            data_json = json.loads(l)
            data_temp = [data_json['author'], data_json['discourse_type'], data_json['text']]
            data.append(data_temp)

        id_num = 0
        for i in range(len(data)):
            for j in range(i+1, len(data)):
                if data[i][1] != data[j][1]:
                    id_num += 1
                    new_id = 'new_id_' + str(id_num)
                    dic_pairs = {'id': new_id,
                                 'discourse_types': [data[i][1], data[j][1]],
                                 'pair': [data[i][2], data[j][2]]
                            }
                    f_pairs.write(json.dumps(dic_pairs) + '\n')
                    if data[i][0] == data[j][0]:
                        dic_truth = {'id': new_id,
                                     'same':True,
                                     'authors':[data[i][0], data[j][0]]
                                     }
                        f_truth.write(json.dumps(dic_truth) + '\n')
                    elif data[i][0] != data[j][0]:
                        dic_truth = {'id': new_id,
                                     'same':False,
                                     'authors':[data[i][0], data[j][0]]
                                     }
                        f_truth.write(json.dumps(dic_truth) + '\n')

    f_truth.close()
    f_pairs.close()


def count_positive_negative(new_truth_path):
    with open(new_truth_path, 'r', encoding='utf-8') as f:
        count = [0,0]
        for l in tqdm(f):
            data = json.loads(l)
            same = data['same']
            if same == True:
                count[0] += 1
            elif same == False:
                count[1] += 1
        print(count)






if __name__ == '__main__':
    text_path = r'D:\NLP\数据集\PAN2022数据集\process_data\new_traindata_text.jsonl'
    truth_path = r'D:\NLP\数据集\PAN2022数据集\process_data\new_traindata_truth.jsonl'
    # datas = get_origin_data(truth_path, text_path)
    # create_author_file(datas, r'D:\NLP\数据集\PAN2022数据集\process_data\author_dt_sen.jsonl')
    new_truth_path = r'D:\NLP\数据集\PAN2022数据集\process_data\new_diy_truth.jsonl'
    new_pairs_path = r'D:\NLP\数据集\PAN2022数据集\process_data\new_diy_pairs.jsonl'
    # create_new_data(new_truth_path, new_pairs_path)
    count_positive_negative(new_truth_path)
