from tensorflow.keras.models import  Model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator, text_segmentate, to_array
from tensorflow.keras.layers import Dropout, Dense, Lambda, LSTM, GlobalAveragePooling1D
import numpy as np
from tqdm import tqdm
from bert4keras.models import build_transformer_model
import tensorflow.python.util.deprecation as deprecation
from pan22_model1_train import get_data
from pan22_model1_bert_model import build_bert_model
from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession

# 此3句代码用于解决NotFoundError: No algorithm worked! when using Conv2D的报错
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

deprecation._PRINT_DEPRECATION_WARNINGS = False

class_nums = 2
maxlen = 256
batch_size = 30

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
            for index, sent in enumerate(text1[:3]):
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
data = get_data(truth_path, text_path, 'short')
train_data_len = int(len(data) * 0.85)
train_generator = data_generator(data[:train_data_len], batch_size)
test_generator = data_generator(data[train_data_len:], batch_size)


model = build_bert_model(config_path ,checkpoint_path, class_nums)
model.load_weights('checkpoint1/best_model.weights')  #加载微调后的权重

cls_layer = Model(inputs=model.input, outputs=model.layers[-2].output)  #后面会从cls_layer提取cls向量作为句子表示

def get_sents_represent(generator, texts_num):
    sents_r = np.array([])
    for x_true, y_true in tqdm(generator):
        sent_r = cls_layer.predict(x_true)
        sents_r = np.append(sents_r, sent_r)
    sents_rs = sents_r.reshape(texts_num, 3, 768)

    return sents_rs


train_sents_rs = get_sents_represent(train_generator, len(data[:train_data_len]))
test_sents_rs = get_sents_represent(test_generator, len(data[train_data_len:]))

np.save("checkpoint1/train_sents_rs.npy",train_sents_rs)
np.save("checkpoint1/test_sents_rs.npy",test_sents_rs)

