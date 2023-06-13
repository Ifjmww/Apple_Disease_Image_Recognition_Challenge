# Apple_Disease_Image_Recognition_Challenge
2023讯飞开放平台-AI开发者大赛-苹果病害图像识别挑战赛

## [比赛链接](https://challenge.xfyun.cn/topic/info?type=apple-diseases)
## 最优模型 EfficientNet-B2  准确率 0.93892

# Train
## step 1  dataset  preprocess (create validiation set)
```
python data_preprocess.py
```

## step 2 train 
### PC
```
python main.py --epochs 100 --batch_size 16 --model_name xxx 
```
### Server
```
python main.py --epochs 200 --batch_size 64 --model_name xxx 
```

## Test (同样需要手动指定所要预测的模型)
```
python test.py --model_name xxx  --mode test
```
