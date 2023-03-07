# pan22_author_verification
## 代码原论文：Authorship verification Based On Fully Interacted Text Segments 
## 论文链接：https://ceur-ws.org/Vol-3180/paper-201.pdf

### 运行环境
#### python 3.7.9
#### tensorflow 2.2.0
#### h5py 2.10
#### numpy 1.19.2
#### bert4keras 0.11.3

### 文件夹说明
1. 2021为2021年作者识别方案
2. baseline为2022年clef组织方提供的两个baseline算法，以及评价代码      
3. proposal 1~5是不同的方案  
4. test中为将pytorch权重文件转换为tensorflow权重文件的代码  
5. collection.py用于分析数据集  
6. enlarge_data.py用于打乱数据，并重新组合，进行数据扩充  
7. pan22_开头的文件为2022年参赛代码  

### 运行顺序及说明
1. pan22_model1_train.py	精调BERT模型  
2. pan22_model1_get_represent.py	用精调后的BERT模型提取文本表示  
3. pan22_save_final_model.py	用提取的文本表示训练decider部分  
4. pan22_main.py	用训练完成的模型生成答案  
5. pan22_verif_evaluator.py	评价函数  
	
