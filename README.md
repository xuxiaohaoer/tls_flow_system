# MWM
词向量建模方法
## Folders:
The repository is organised as follows:
* `data_raw/`原始流量数据
* `data_feature/`提取好的词向量特征
* `data_process/`特征预处理
* `model_save/`模型保存
* `model_train/`模型训练
* `setting.yml`全局配置文件
## Data:
* 原始数据放在data_raw里边
* 处理好的词向量特征放在data_featurea里边
## Models:
* 词向量建模
* Multihead Attention
* BiLSTM
* 训练好的模型放在model_train中
## Run:
* `python3 data_process.data_word_flow.py #特征提取`
* `python3 model_train.DS.py #模型训练`
## Ohter：

