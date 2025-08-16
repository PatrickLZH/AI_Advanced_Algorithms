# 导包
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class_nums = 3
batch_size = 128
learning_rate = 0.001
training_epochs = 15

def _norm_advprop(img):
    return img * 2.0 - 1.0

def build_transform(dest_image_size):
    normalize = transforms.Lambda(_norm_advprop)
    if not isinstance(dest_image_size, tuple):
        dest_image_size = (dest_image_size, dest_image_size)
    else:
        dest_image_size = dest_image_size
    transform = transforms.Compose([
        transforms.Resize(dest_image_size, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize
    ])
    return transform

def build_data_set(dest_image_size, data):
    transform = build_transform(dest_image_size)
    dataset = datasets.ImageFolder(data, transform=transform, target_transform=None)
    return dataset

def evaluate_model(model, data_loader, device):
    """
    在给定数据集上评估模型性能
    """
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 禁用梯度计算
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy


def visualize_predictions(model, data_loader, device, class_names, num_images=16):
    """
    可视化模型预测结果
    """
    print("开始可视化预测结果...")
    model.eval()
    
    # 存储图像、标签和预测结果
    images_so_far = 0
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            print(f"处理批次 {i+1}, 批次大小: {images.size()[0]}")
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            for j in range(images.size()[0]):
                images_so_far += 1
                
                # 当达到指定数量时创建新图形
                if (images_so_far - 1) % 16 == 0:
                    if images_so_far > 1:
                        plt.tight_layout()
                        plt.show()
                    fig = plt.figure(figsize=(15, 10))
                    print(f"显示第 {images_so_far} 到 {min(images_so_far + 15, num_images)} 张图片")
                
                ax = plt.subplot(4, 4, (images_so_far - 1) % 16 + 1)
                ax.axis('off')
                
                # 反归一化图像
                img = images.cpu().data[j]
                img = (img + 1) / 2  # 从 [-1, 1] 转换回 [0, 1]
                img = np.clip(img.numpy().transpose((1, 2, 0)), 0, 1)
                
                # 获取预测概率
                prob = probs[j][preds[j]].item()
                
                # 设置标题显示真实标签和预测标签
                title_color = 'black' if preds[j] == labels[j] else 'red'
                ax.set_title(f'真实: {class_names[labels[j]]} 预测: {class_names[preds[j]]} 概率: {prob:.2f}',
                           color=title_color, fontsize=10)
                
                plt.imshow(img)
                
                if images_so_far == num_images or images_so_far == len(data_loader.dataset):
                    plt.tight_layout()
                    plt.show()
                    print("可视化完成，图像已显示")
                    return

# 数据处理
train_data = build_data_set(224, './data/train')
Val_data = build_data_set(224, './data/val')
test_data = build_data_set(224, './data/test')

train_loader = DataLoader(dataset=train_data,
                          batch_size=batch_size,
                          shuffle=True)
val_loader = DataLoader(dataset=Val_data,
                        batch_size=batch_size,
                        shuffle=False)
test_loader = DataLoader(dataset=test_data,
                         batch_size=batch_size,
                         shuffle=False)

# 获取类别名称
class_names = train_data.classes
print(f"类别名称: {class_names}")

# 检查是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 检查本地是否已有模型文件
local_model_path = 'D:\\MyProject\\Model\\resnet50\\resnet50_pretrained.pth'
os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
if os.path.exists(local_model_path):
    # 从本地加载
    print("Loading ResNet50 from local file...")
    resnet = models.resnet50()
    resnet.load_state_dict(torch.load(local_model_path))
else:
    # 从网络下载并保存到本地
    print("Downloading ResNet50 and saving to local file...")
    resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    torch.save(resnet.state_dict(), local_model_path)

# 调整ResNet的最后一个全连接层
resnet.fc = nn.Linear(resnet.fc.in_features, class_nums)
resnet = resnet.to(device)

# 模型训练
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet.parameters(), lr=learning_rate, momentum=0.9)

for epoch in range(training_epochs):
    total_loss = 0
    total_accuracy = 0
    num_batches = 0

    for i, (batch_xs, y) in enumerate(train_loader):
        batch_xs = batch_xs.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        hypothesis = resnet(batch_xs)
        loss = criterion(hypothesis, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()  # 累积损失
        # 计算训练准确率
        accuracy = (torch.max(hypothesis, dim=1)[1] == y).float().mean()
        total_accuracy += accuracy.item()
        num_batches += 1
    # 计算平均值
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    # 计算验证集准确率
    val_accuracy = evaluate_model(resnet, val_loader, device)

    print(f'Epoch:{epoch+1}/{training_epochs} loss:{avg_loss:.4f} accuracy:{avg_accuracy:.4f} val_accuracy:{val_accuracy:.4f}')

# 训练完成后在测试集上评估
print("Training completed. Evaluating on test set...")
test_accuracy = evaluate_model(resnet, test_loader, device)
print(f'Test Accuracy: {test_accuracy:.4f}')

# 可视化测试结果
print("Visualizing test results...")
visualize_predictions(resnet, test_loader, device, class_names)