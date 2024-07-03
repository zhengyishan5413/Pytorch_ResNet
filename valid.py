import os.path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from model import ResNet18  # 假设 ResNet18 定义在 model.py 中
from tqdm import tqdm

# 数据转换和加载测试数据集
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

data_folder = './data_train_val/val'
test_dataset = datasets.ImageFolder(root=data_folder, transform=data_transform)

batch_size = 32
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model_name = "ResNet18"
model = ResNet18(num_classes=26)

# 加载预训练模型权重
model_weights_path = f'./weights/{model_name}.pth'
model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))  # 若使用GPU，去掉 map_location 参数
model.eval()

# 在测试集上进行预测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

all_labels = []
all_predictions = []
all_inputs = []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc='Validation', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        all_inputs.extend(inputs.cpu().numpy())

# 计算混淆矩阵
conf_matrix = confusion_matrix(all_labels, all_predictions)

# 绘制混淆矩阵热力图
plt.figure(figsize=(19, 12))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes,
            yticklabels=test_dataset.classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=35)
os.makedirs("./images_final", exist_ok=True)
plt.savefig(f"./images_final/Confusion_{model_name}.png")
plt.close()  # 关闭 plt.show()，防止阻塞代码运行

# 输出混淆矩阵
print("Confusion Matrix:")
print(conf_matrix)

# 计算并输出分类报告
classification_rep = classification_report(all_labels, all_predictions, target_names=test_dataset.classes)
print("\nClassification Report:")
print(classification_rep)

# 计算并输出准确率、精确率、召回率和F1值
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average='macro')
recall = recall_score(all_labels, all_predictions, average='macro')
f1 = f1_score(all_labels, all_predictions, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# 找出错误分类的样本
misclassified_indices = [i for i, (true, pred) in enumerate(zip(all_labels, all_predictions)) if true != pred]
print(f"\nNumber of misclassified samples: {len(misclassified_indices)}")

# 打印部分错误分类的样本数据
print("\nMisclassified samples (showing up to 10 samples):")
for i in misclassified_indices[:10]:
    true_label = test_dataset.classes[all_labels[i]]
    predicted_label = test_dataset.classes[all_predictions[i]]
    print(f"Sample {i}: True Label = {true_label}, Predicted Label = {predicted_label}")

    # 显示图片 (可选)
    plt.imshow(all_inputs[i].transpose((1, 2, 0)))  # 转置以使维度与 matplotlib 兼容
    plt.title(f'True: {true_label}, Predicted: {predicted_label}')
    plt.show()

# 可以保存一些错误分类的图片用于进一步分析
os.makedirs("./misclassified_samples", exist_ok=True)
for i in misclassified_indices[:10]:
    true_label = test_dataset.classes[all_labels[i]]
    predicted_label = test_dataset.classes[all_predictions[i]]
    plt.imsave(f"./misclassified_samples/sample_{i}_true_{true_label}_pred_{predicted_label}.png",
               all_inputs[i].transpose((1, 2, 0)))
