import json, os, re
import pandas as pd
from nltk.tokenize import word_tokenize


def extrac_list(list_src, line_no):
    res1 = []
    for i in list_src:
        res1.append(i[line_no])
    return res1


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


def author_sen(list1, list2):
    auth_list = []
    essay_list = []
    mail_list = []
    mess_list = []
    memo_list = []
    sen_list = []
    for i in range(len(list1)):
        data1 = json.loads(list1[i])
        data2 = json.loads(list2[i])
        discourse_type = data1["discourse_types"]   # DT对
        pair = data1["pair"]                        # 文本对
        same = data2["same"]                        # 真值
        authors = data2["authors"]                  # 作者对
        for j in range(2):
            if authors[j] not in auth_list:
                auth_list.append(authors[j])
                if discourse_type[j] == "essay":
                    essay_list.append([pair[j]])
                else:
                    essay_list.append([])
                if discourse_type[j] == "email":
                    mail_list.append([pair[j]])
                else:
                    mail_list.append([])
                if discourse_type[j] == "text_message":
                    mess_list.append([pair[j]])
                else:
                    mess_list.append([])
                if discourse_type[j] == "memo":
                    memo_list.append([pair[j]])
                else:
                    memo_list.append([])
                sen_list.append([pair[j]])
            elif authors[j] in auth_list:
                author_pos = auth_list.index(authors[j])
                if discourse_type[j] == "essay":
                    essay_list[author_pos].append(pair[j])
                if discourse_type[j] == "email":
                    mail_list[author_pos].append(pair[j])
                if discourse_type[j] == "text_message":
                    mess_list[author_pos].append(pair[j])
                if discourse_type[j] == "memo":
                    memo_list[author_pos].append(pair[j])
                sen_list[author_pos].append(pair[j])
    auth_sen_list_desperate = []
    auth_sen_list = []
    for i in range(len(auth_list)):
        auth_sen_list_desperate.append([auth_list[i], essay_list[i], mail_list[i], mess_list[i], memo_list[i]])
    for i in range(len(auth_list)):
        auth_sen_list.append([auth_list[i], sen_list[i]])
    return auth_sen_list_desperate, auth_sen_list


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


def sen_of_dt(auth_sen_list):
    essay_lis = []
    email_lis = []
    message_lis = []
    memo_lis = []
    for lis in auth_sen_list:
        for sen1 in lis[1]:
            essay_lis.append(sen1)
        for sen2 in lis[2]:
            email_lis.append(sen2)
        for sen3 in lis[3]:
            message_lis.append(sen3)
        for sen4 in lis[4]:
            memo_lis.append(sen4)
    return essay_lis, email_lis, message_lis, memo_lis


def count_by_ref(str_list, ref_angular, ref_special):
    count_list_angular = [0] * len(ref_angular)
    count_list_special = [0] * len(ref_special)
    angular_list_count = count_angular_symbol(str_list)
    special_list_count = count_special_symbol(str_list)
    an_list = extrac_list(angular_list_count, 0)
    an_count = extrac_list(angular_list_count, 1)
    sp_list = extrac_list(special_list_count, 0)
    sp_count = extrac_list(special_list_count, 1)
    for an in ref_angular:
        if an in an_list:
            count_list_angular[ref_angular.index(an)] = an_count[an_list.index(an)]
    for sp in ref_special:
        if sp in sp_list:
            count_list_special[ref_special.index(sp)] = sp_count[sp_list.index(sp)]

    return count_list_angular, count_list_special


def separate_into_words(str_src, ref_angular, ref_special):
    # 首先去除文本中的各类特殊符号
    str_temp = str_src
    for sym in ref_angular:
        if sym.strip() in str_temp:
            str_temp = str_temp.replace(sym.strip(), '')
    for sym in ref_special:
        if sym.strip() in str_temp:
            str_temp = str_temp.replace(sym.strip(), '')
    word_list = word_tokenize(str_temp)
    return len(word_list)


def count_in_step(sen_lis, step):
    start = 0
    end = start + step
    max_len = max(sen_lis)
    list_len = 4000 // step
    sen_len_step = [0] * list_len
    list_index = 0
    while end <= (max_len+step):
        count = 0
        for i in sen_lis:
            if (i >= start) and (i < end):
                count += 1
        sen_len_step[list_index] = count
        start = start + step
        end = end + step
        list_index += 1
    return sen_len_step


def get_sen_len_list(sen_list, angular_list, special_list):
    sen_len_list = []                               # 获取所有文本的长度
    no = 0
    for i in sen_list:
        no += 1
        if no % 1000 == 0:
            print(no)
        sen_len_list.append(separate_into_words(i, angular_list, special_list))

    sen_len_list = sorted(sen_len_list)

    len_list_50 = count_in_step(sen_len_list, 50)
    len_list_100 = count_in_step(sen_len_list, 100)
    len_list_150 = count_in_step(sen_len_list, 150)
    len_list_200 = count_in_step(sen_len_list, 200)
    len_list_250 = count_in_step(sen_len_list, 250)

    # print(len_list_50)
    # print(len_list_100)
    # print(len_list_150)
    # print(len_list_200)
    # print(len_list_250)

    for i in len_list_50:
        print(i, end="\t")
    print()
    for i in len_list_100:
        print(i, end="\t")
    print()
    for i in len_list_150:
        print(i, end="\t")
    print()
    for i in len_list_200:
        print(i, end="\t")
    print()
    for i in len_list_250:
        print(i, end="\t")
    print()
    for i in range(100):
        print('-', end='')
    print()




if __name__ == '__main__':
    dir_path = r'D:\NLP\数据集\PAN2022数据集\pan22-authorship-verification-training-dataset'
    pairs = load_text_data(dir_path + r'\pairs.jsonl')
    truth = load_text_data(dir_path + r'\truth.jsonl')
    dt_list, max_len, min_len, sen_list = filter_info_pairs(pairs)
    author_count_list, author_count, label_count = filter_info_truth(truth)
    print("discourse types: ", dt_list)                 # 各种DT及计数
    # print("max text length: ", max_len)                 # 最大文本字符长度,22160
    # print("min text length: ", min_len)                 # 最小文本字符长度,230
    # print("author list and count: ", author_count_list) # 作者列表，及每个作者文本数
    # print("author count: ", author_count)               # 作者数量,56
    # print("true examples: ", label_count[0])            # 正例数量,6132
    # print("false examples: ", label_count[1])           # 反例数量,6132

    print(count_angular_symbol(sen_list))               # 尖括号符号及数量
    angular_symbol_list = count_angular_symbol(sen_list)    # 尖括号符号及数量
    print("\nangular_symbol\t" + "count")
    for i in angular_symbol_list:
        print(i[0], end="\t")
        print(i[1])

    print(count_special_symbol(sen_list))               # 特殊符号及数量
    special_symbol_list = count_special_symbol(sen_list)    # 特殊符号及数量
    print("\nspecial_symbol\t" + "count")
    for i in special_symbol_list:
        print(i[0], end="\t")
        print(i[1])

    print("\nessay\t" + "email\t" + "text_message\t" + "memo\t" + "total")  # 各作者四种DT文本量
    author_sen_desperate, author_sen_list = author_sen(pairs, truth)
    for i in author_sen_desperate:
        print(author_sen_desperate.index(i) + 1,end="\t")
        print(i[0],end='\t')
        print(len(i[1]),end='\t')
        print(len(i[2]),end='\t')
        print(len(i[3]),end='\t')
        print(len(i[4]),end='\t')
        print(len(i[1])+len(i[2])+len(i[3])+len(i[4]))

    essay_list, email_list, msg_list, memo_list = sen_of_dt(author_sen_desperate)    # 四种DT的所有句子

    angular_list = []                   # 所有尖括号符号的列表
    for i in angular_symbol_list:
        angular_list.append(i[0])
    special_list = []                   # 所有特殊符号的列表
    for i in special_symbol_list:
        special_list.append(i[0])

    # 四种DT中，尖括号和特殊符号的数量
    an_of_essay, sp_of_essay = count_by_ref(essay_list, angular_list, special_list)
    an_of_email, sp_of_email = count_by_ref(email_list, angular_list, special_list)
    an_of_msg, sp_of_msg = count_by_ref(msg_list, angular_list, special_list)
    an_of_memo, sp_of_memo = count_by_ref(memo_list, angular_list, special_list)

    author_list = extrac_list(author_sen_list, 0)   # 作者列表

    # get_sen_len_list(sen_list, angular_list, special_list)  # 获取所有句子的长度

    print("essay_list")
    get_sen_len_list(essay_list, angular_list, special_list)
    print("email_list")
    get_sen_len_list(email_list, angular_list, special_list)
    print("msg_list")
    get_sen_len_list(msg_list, angular_list, special_list)
    print("memo_list")
    get_sen_len_list(memo_list, angular_list, special_list)



'''
    # 每个作者的尖括号和特殊符号的数量
    for i in author_sen_list:
        print(i[0], end="\t")
        sen = i[1]
        an_of_sen, sp_of_sen = count_by_ref(sen, angular_list, special_list)
        for j in an_of_sen:
            print(str(j), end="\t")
        print()
        for j in sp_of_sen:
            print(str(j), end="\t")
        print()
'''

'''
    for i in an_of_essay:
        print(str(i), end="\t")
    print()
    for i in an_of_email:
        print(str(i), end="\t")
    print()
    for i in an_of_msg:
        print(str(i), end="\t")
    print()
    for i in an_of_memo:
        print(str(i), end="\t")
    print()
    for i in sp_of_essay:
        print(str(i), end="\t")
    print()
    for i in sp_of_email:
        print(str(i), end="\t")
    print()
    for i in sp_of_msg:
        print(str(i), end="\t")
    print()
    for i in sp_of_memo:
        print(str(i), end="\t")
    print()
'''
