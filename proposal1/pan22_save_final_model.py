import numpy as np

from pan22_model2_bert_model import get_data

text_path = r'D:\NLP\数据集\PAN2022数据集\process_data\new_train.jsonl'
truth_path = r'D:\NLP\数据集\PAN2022数据集\process_data\new_train_truth.jsonl'
data_short = get_data(truth_path, text_path, 'short')
data_long = get_data(truth_path,text_path, 'long')
train_data_len_1 = int(len(data_short) * 0.85)
train_data_len_2 = int(len(data_long) * 0.85)

labels_1=[]
for i in data_short:
    labels_1.append(i[2])

labels_2=[]
for i in data_long:
    labels_2.append(i[2])

train_sents_r_1 = np.load("checkpoint1/train_sents_rs.npy",mmap_mode = 'r')
train_labels_1=np.array(labels_1[:train_data_len_1])
valid_sents_r_1 = np.load("checkpoint1/test_sents_rs.npy",mmap_mode = 'r')
valid_labels_1=np.array(labels_1[train_data_len_1:])

train_sents_r_2 = np.load("checkpoint2/train_sents_rs.npy",mmap_mode = 'r')
train_labels_2=np.array(labels_2[:train_data_len_2])
valid_sents_r_2 = np.load("checkpoint2/test_sents_rs.npy",mmap_mode = 'r')
valid_labels_2=np.array(labels_2[train_data_len_2:])

train_sents_r_1 = np.mean(train_sents_r_1, axis=1, keepdims=True)
train_sents_r_2 = np.mean(train_sents_r_2, axis=1, keepdims=True)
valid_sents_r_1 = np.mean(valid_sents_r_1, axis=1, keepdims=True)
valid_sents_r_2 = np.mean(valid_sents_r_2, axis=1, keepdims=True)

from tensorflow.keras.layers import Dense,GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

callbacks = [

    ModelCheckpoint(
        f'weights1.h5',
        monitor='val_sparse_categorical_accuracy',
        save_weights_only=True,
        save_best_only=True,
        verbose=1,
        mode='max'),
]

model = Sequential()
model.add(GlobalAveragePooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dense(units=2, activation='softmax', ))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
model.fit(train_sents_r_2,train_labels_2,epochs=500,validation_data=(valid_sents_r_2,valid_labels_2),callbacks=callbacks)
model.fit(train_sents_r_1,train_labels_1,epochs=500,validation_data=(valid_sents_r_1,valid_labels_1),callbacks=callbacks)

model.save('final_model.h5')