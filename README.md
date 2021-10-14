# WVM
词向量建模方法
## Folders:
The repository is organised as follows:
* `data_raw/`原始流量数据
* `data_cut/`切割后的流量数据
* `data_feature/`提取好的词向量特征
* `data_process/`特征预处理
* `flow_cut/`流切割模块
* `model_save/`模型保存模块
* `model_train/`模型训练模块
* `setting.yml`全局配置文件
## Data:
* 原始数据放在`data_raw/`里边
* 切割好的数据放在`data_cut/tls`中,请分别建立`data_cut/test/black`,`data_cut/test/white`,`data_cut/train/black`, `data_cut/train/white`，根据训练与测试与黑白标签放入不同的文件夹中
* 处理好的词向量特征放在data_featurea里边，对应`f_data_word`
## Models:
* 词向量建模
* Multihead Attention
* BiLSTM
* 训练好的模型放在model_train中
## Run:
直接在文件目录下运行以下代码即可
* `python3 flow_cut/pre.py #流量切割`
* `python3 data_process/data_word_flow.py #特征提取`
* `python3 model_train/DS.py #模型训练`
## Other：

