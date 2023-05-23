import os
import csv
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.transforms import transforms
from utils.tools import choose_model
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_paths = os.listdir(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_paths[idx])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, 0  # 设置标签为0或任何其他适合的数值


# 测试集路径和保存结果的CSV文件路径
test_data_dir = "./dataset/test/"
EXP = 'mobilenetv3_large'
model_name = 'mobilenetv3_large'
output_csv_path = "./results/" + EXP + '/output.csv'
num_classes = 9

if not os.path.exists("./results/" + EXP + '/'):
    os.makedirs("./results/" + EXP + '/')

# 数据预处理
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载测试集数据
test_dataset = CustomDataset(data_dir=test_data_dir, transform=data_transforms)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 加载最优模型
model = choose_model(model_name, num_classes)
model.load_state_dict(torch.load("./models/" + EXP + "/best_model.pth"))  # 加载最优模型权重

# 将模型放在GPU上（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 预测并保存结果
results = []
model.eval()
with torch.no_grad():
    for inputs, _ in tqdm(test_dataloader, desc="[Pred]"):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        results.append(preds.item())

# 写入CSV文件
with open(output_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['uuid', 'label'])
    for idx, image_path in enumerate(test_dataset.image_paths):
        writer.writerow([image_path, results[idx] + 1])
