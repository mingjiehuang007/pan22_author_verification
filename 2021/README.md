# task-for-authorship-verification
任务：small_dataset有52601对大文本文档，目标是根据作者写作风格来判断一对文档是否由同一个作者所写，属于二分类任务。
本项目代码运行步骤如下：

1.preprocess_and_finetune_model.py

2.get_sentences_represent.py

3.save_final_model.py

4.main.py

最后利用pan20_verif_evaluator.py来得到测试数据集的性能指标

bert4keras==0.9.8
keras==2.3.1
tensorflow==1.15.2
python3.7
