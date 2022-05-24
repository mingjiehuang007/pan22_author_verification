import numpy as np

from preprocess_and_finetune_model import get_data, truth_path, text_path
data = get_data(truth_path,text_path)

labels=[]
for i in data:
    labels.append(i[2])

train_sents_r = np.load("train_sents_rs.npy",mmap_mode = 'r')
train_labels=np.array(labels[:8585])
valid_sents_r = np.load("test_sents_rs.npy",mmap_mode = 'r')
valid_labels=np.array(labels[8585:])


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
model.add(Dense(32, activation='relu', ))
model.add(Dense(units=2, activation='softmax', ))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
model.fit(train_sents_r,train_labels,epochs=500,validation_data=(valid_sents_r,valid_labels),callbacks=callbacks)
model.save('final_model.h5')