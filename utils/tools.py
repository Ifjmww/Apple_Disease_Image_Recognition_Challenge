from torchvision.models import resnet50
from torchvision.models import mobilenet_v3_small
from torchvision.models import mobilenet_v3_large
import torch.nn as nn
import torch
from networks.GhostResNet import ghostresnet50


def choose_model(model_name, num_classes):
    if model_name == 'resnet50':
        print("choose resnet-50")
        model = resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        return model
    elif model_name == 'mobilenetv3_small':
        print("choose mobilenetv3_small")
        model = mobilenet_v3_small(pretrained=True)
        num_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_features, num_classes)
        return model
    elif model_name == 'mobilenetv3_large':
        print("choose mobilenetv3_large")
        model = mobilenet_v3_large(pretrained=True)
        num_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_features, num_classes)
        return model
    elif model_name == 'ghostnet':
        print("choose ghostnet")
        model = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)  # 修改分类器的输出大小为9类
        return model
    elif model_name == 'ghost-resnet':
        print('choose ghost-resnet')
        model = ghostresnet50(num_classes=9)
        return model
