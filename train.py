import torch
import models

import get_data
from torch.utils.tensorboard import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.ModelSimple().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

# 添加TensorBoard记录器
writer = SummaryWriter('./runs/sentiment')

loss_func = torch.nn.CrossEntropyLoss() # 交叉熵损失函数

token_list = get_data.token_list
labels = get_data.labels

dev_list = get_data.dev_list
dev_labels = get_data.dev_labels

batch_size = 128
train_length = len(labels)
for epoch in (range(21)):
    train_num = train_length // batch_size
    train_loss, train_correct = 0, 0
    for i in (range(train_num)):
        start = i * batch_size
        end = (i + 1) * batch_size

        batch_input_ids = torch.tensor(token_list[start:end]).to(device)
        batch_labels = torch.tensor(labels[start:end]).to(device)

        pred = model(batch_input_ids)

        loss = loss_func(pred, batch_labels.type(torch.uint8))

        optimizer.zero_grad() # 梯度清零
        loss.backward() # 计算误差
        optimizer.step() # 反向传播，更新参数

        train_loss += loss.item() # 计算损失
        train_correct += ((torch.argmax(pred, dim=-1) == (batch_labels)).type(torch.float).sum().item() / len(batch_labels))

    train_loss /= train_num
    train_correct /= train_num
    print("train_loss:", train_loss, "train_correct:", train_correct)
    
    # 记录训练数据到TensorBoard
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_correct, epoch)

    test_pred = model(torch.tensor(dev_list).to(device))
    correct = (torch.argmax(test_pred, dim=-1) == (torch.tensor(dev_labels).to(device))).type(torch.float).sum().item() / len(test_pred)
    print("test_acc:",correct)
    
    # 记录测试准确率到TensorBoard
    writer.add_scalar('Accuracy/test', correct, epoch)
    print("-------------------")

# 训练结束后关闭TensorBoard记录器
writer.close()



