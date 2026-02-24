# Chinese-text-sentiment-classification-based-on-BERT-fine-tuning

基于 BERT 微调的中文文本情感分类：该项目基于预训练模型BERT开发，实现了中文文本的二分类情感分析任务。项目采用迁移学习策略，在BERT的预训练权重基础上，通过添加全连接层并冻结上游模型参数的方式进行微调，以降低计算成本。数据处理模块使用BertTokenizer对文本进行编码和填充；模型架构在BERT输出的CLS标记的基础上进行分类，通过Softmax激活函数输出概率分布；训练采用交叉熵损失函数和Adam优化器，在ChnSentiCorp数据集上训练，实现了92.35%的分类精度。

## 文件介绍

ChnSentiCorp.txt：数据集

get_data.py：用于读取数据

models.py：模型相关代码

train.py：用于训练模型

## 快速开始

运行train.py

命令行输入：`tensorboard --logdir=./runs`以查看训练结果。
