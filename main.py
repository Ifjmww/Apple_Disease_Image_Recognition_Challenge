import argparse
from test import test
from train import train
from utils.tools import save_args_info


def add_args():
    parser = argparse.ArgumentParser(description="APPLE")

    parser.add_argument("--epochs", default=200, type=int, help="max number of training epochs")
    parser.add_argument("--seed", default=42, type=int, help="torch.manual_seed")
    parser.add_argument("--device", default='cuda:0', type=str, help="GPU id for training")
    parser.add_argument("--mode", default='train', choices=['train', 'test'], type=str, help='network run mode')
    parser.add_argument("--batch_size", default=16, type=int, help="number of batch size")
    parser.add_argument("--batch_size_test", default=1, type=int, help="number of batch size when not training")
    parser.add_argument("--train_path", default="./dataset/train/", type=str, help="dataset directory")
    parser.add_argument("--valid_path", default="./dataset/valid/", type=str, help="dataset directory")
    parser.add_argument("--test_path", default="./dataset/test/", type=str, help="dataset directory")
    parser.add_argument("--num_classes", default=9, type=int, help="number of classes")
    parser.add_argument("--model_name", default="resnet50",
                        choices=['resnet50', 'mobilenetv3_small', 'mobilenetv3_large', 'ghostnet', 'ghost-resnet', 'ghostnetv2', 'enet-b0', 'enet-b1',
                                 'enet-b2''enet-b3'],
                        type=str, help="model name")
    parser.add_argument("--save_every", default=10, type=int, help="validation frequency")
    parser.add_argument("--lr", default=0.001, type=float, help="optimization learning rate")

    args = parser.parse_args()

    print()
    print(">>>============= args ====================<<<")
    print()
    print(args)  # print command line args
    print()
    print(">>>=======================================<<<")

    return args


def main(args):
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    else:
        raise ValueError("Only ['train', 'test'] mode is supported.")
    save_args_info(args)


if __name__ == "__main__":
    args = add_args()
    main(args)
