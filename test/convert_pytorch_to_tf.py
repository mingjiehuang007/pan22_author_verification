import torch
import tensorflow as tf

def convert(bin_path, ckptpath):
    with tf.compat.v1.Session() as sess:
        for var_name, value in torch.load(bin_path, map_location='cpu').items():
            print(var_name)
            tf.compat.v1.Variable(initial_value=value, name=var_name)
        saver = tf.compat.v1.train.Saver()
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, ckptpath)

bin_path = 'pytorch_model.bin'
ckpt_path = 'bert_model.ckpt'
convert(bin_path, ckpt_path)