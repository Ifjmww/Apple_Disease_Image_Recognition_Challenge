import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.tools import choose_model


def train(args):
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    # 数据集路径和超参数
    weights = torch.tensor([34.9273, 35.3809, 4.5864, 42.7905, 28.1827, 39.2932, 12.3087, 5.2968, 2.6973]).to(device)

    # 数据预处理
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载训练集和验证集
    train_dataset = ImageFolder(args.train_path, transform=data_transforms)
    val_dataset = ImageFolder(args.valid_path, transform=data_transforms)

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    if not os.path.exists('./models/' + args.model_name + '/'):
        os.makedirs('./models/' + args.model_name + '/')

    # 加载预训练的ResNet-50模型
    model = choose_model(args.model_name, args.num_classes)

    # 将模型放在GPU上（如果可用）
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 训练和验证过程
    best_val_loss = float('inf')
    best_model_weights = model.state_dict()

    for epoch in range(args.epochs):
        # 训练
        model.train()
        train_loss = 0.0
        train_corrects = 0

        for inputs, labels in tqdm(train_dataloader, desc="Epoch {}/{}".format(epoch + 1, args.epochs)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_corrects += torch.sum(preds == labels.data)

        train_loss = train_loss / len(train_dataloader)
        train_acc = train_corrects / len(train_dataset)

        # 验证
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_dataloader)
        val_acc = val_corrects / len(val_dataset)

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()

        if (epoch + 1) % args.save_every == 0:
            # 每x轮保存一次模型
            torch.save(model.state_dict(), './models/' + args.model_name + '/' + args.model_name + '_epoch_' + str(epoch + 1) + '_acc_' + str(val_acc.item()) + '.pth')

        print("Epoch {}/{}: Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}"
              .format(epoch + 1, args.epochs, train_loss, train_acc, val_loss, val_acc))

    # 加载最优模型权重
    model.load_state_dict(best_model_weights)

    # 保存模型
    torch.save(model.state_dict(), './models/' + args.model_name + '/best_model.pth')
