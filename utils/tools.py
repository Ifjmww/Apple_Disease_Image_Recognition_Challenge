from torchvision.models import resnet50
from torchvision.models import mobilenet_v3_small
from torchvision.models import mobilenet_v3_large
import torch.nn as nn
import torch
from networks.GhostResNet import ghostresnet50
from networks.ghostnetv2 import ghostnetv2
from efficientnet_pytorch import EfficientNet
import os


def save_args_info(args):
    # save args to config.txt
    argsDict = args.__dict__
    result_path = './models/' + args.model_name + '/'

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with open(result_path + 'config.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


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
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    elif model_name == 'ghost-resnet':
        print('choose ghost-resnet')
        model = ghostresnet50(num_classes=num_classes)
        return model
    elif model_name == 'ghostnetv2':
        print('choose ghostnetv2')
        model = ghostnetv2(num_classes)
        return model
    elif model_name == 'enet-b0':
        print('choose efficientnet-b0')
        model = EfficientNet.from_pretrained('efficientnet-b0')
        in_features = model._fc.in_features
        model._fc = nn.Linear(in_features, num_classes)
        return model
    elif model_name == 'enet-b1':
        print('choose efficientnet-b1')
        model = EfficientNet.from_pretrained('efficientnet-b1')
        in_features = model._fc.in_features
        model._fc = nn.Linear(in_features, num_classes)
        return model
    elif model_name == 'enet-b2':
        print('choose efficientnet-b2')
        model = EfficientNet.from_pretrained('efficientnet-b2')
        in_features = model._fc.in_features
        model._fc = nn.Linear(in_features, num_classes)
        return model
    elif model_name == 'enet-b3':
        print('choose efficientnet-b3')
        model = EfficientNet.from_pretrained('efficientnet-b3')
        in_features = model._fc.in_features
        model._fc = nn.Linear(in_features, num_classes)
        return model