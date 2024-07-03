import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import ResNet18  # 假设你有一个定义在 model.py 中的 ResNet18 模型
import time
import os
from tqdm import tqdm
import datetime

# 配置 CUDA 环境
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 配置超参数
BATCH_SIZE = 128
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
MOMENTUM = 0.9
MODEL_NAME = "ResNet18"

# 数据路径
TRAIN_DATA_PATH = './data_train_val/train'
VAL_DATA_PATH = './data_train_val/val'
LOG_DIR = "logs"
WEIGHTS_DIR = './weights'
IMAGES_DIR = './images'

# 创建必要的目录
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


# 数据预处理与加载
def get_data_loaders(train_data_path, val_data_path, batch_size):
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(root=train_data_path, transform=data_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = datasets.ImageFolder(root=val_data_path, transform=data_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
    #分别为训练集和验证集创建ImageFolder数据集和DataLoader，并返回这两个数据加载器。


# 获取数据加载器
train_loader, val_loader = get_data_loaders(TRAIN_DATA_PATH, VAL_DATA_PATH, BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化模型、损失函数和优化器
model = ResNet18(num_classes=26).to(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

# 记录文件名
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
file_train_name = os.path.join(LOG_DIR, f"{MODEL_NAME}_training_results_{current_time}.txt")
file_val_name = os.path.join(LOG_DIR, f"{MODEL_NAME}_validation_results_{current_time}.txt")


# 训练与验证函数
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    for inputs, labels in tqdm(train_loader, desc='Training', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct_predictions / len(train_loader.dataset)
    return epoch_loss, epoch_acc


def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = correct_predictions / len(val_loader.dataset)
    return epoch_loss, epoch_acc


# 模型训练过程
def train_model(num_epochs, model, train_loader, val_loader, criterion, optimizer, device, file_train_name,
                file_val_name):
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    best_acc = 0.0

    #执行整个训练过程，记录每个周期的训练损失、准确率、验证损失、准确率，并在每个周期结束时写入日志文件。
    start_time = time.time()
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        with open(file_train_name, "a") as file:
            file.write(f'{epoch + 1}/{num_epochs} {train_loss:.4f} {train_acc:.4f}\n')

        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        with open(file_val_name, "a") as file:
            file.write(f'{epoch + 1}/{num_epochs} {val_loss:.4f} {val_acc:.4f}\n')

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict()
            print(f"New best model with accuracy {best_acc:.4f}")

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    print(f"Training complete in {time.time() - start_time:.2f} seconds")
    torch.save(best_model_state, os.path.join(WEIGHTS_DIR, f'{MODEL_NAME}.pth'))

    return train_loss_history, train_acc_history, val_loss_history, val_acc_history


# 绘制训练和验证曲线
def plot_history(train_loss_history, train_acc_history, val_loss_history, val_acc_history, num_epochs, model_name):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_loss_history, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_loss_history, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_acc_history, label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1), val_acc_history, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, f'{model_name}.png'), dpi=300)
    plt.show()


# 主程序
if __name__ == "__main__":
    train_loss_history, train_acc_history, val_loss_history, val_acc_history = train_model(
        NUM_EPOCHS, model, train_loader, val_loader, criterion, optimizer, device,
        file_train_name, file_val_name
    )

    plot_history(train_loss_history, train_acc_history, val_loss_history, val_acc_history, NUM_EPOCHS, MODEL_NAME)
    #调用train_model函数开始训练流程。
    #训练完成后，调用plot_history展示并保存训练与验证结果的图表。