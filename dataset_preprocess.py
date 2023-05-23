import os
import shutil
import numpy as np


def divide_dataset(train_path):
    divide_rate = 0.20

    for cls in os.listdir(train_path):
        # ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9']
        val_path = './dataset/valid/' + cls + '/'
        if not os.path.exists(val_path):
            os.makedirs(val_path)

        train_list = os.listdir(train_path + cls + '/')
        number = int(divide_rate * len(train_list))

        print('divide dataset number is:', number)
        index = np.random.choice(len(train_list), number, replace=False)

        for i in range(number):
            print(cls, ': ', train_list[index[i]])
            shutil.move(os.path.join(train_path + cls + '/', train_list[index[i]]), val_path)


def main():
    # 将原始训练集划分出一个验证集
    train_path = './dataset/train/'
    divide_dataset(train_path)


if __name__ == '__main__':
    main()
