# Apple_Disease_Image_Recognition_Challenge
2023讯飞开放平台-AI开发者大赛-苹果病害图像识别挑战赛

# Train
## step 1  dataset  preprocess (create validiation set)
```
python data_preprocess.py
```

## step 2 train (训练前需手动在代码中设置超参数，以及所要使用的模型，因为这里偷懒，没有写args参数👻)
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
python test.py --model_name xxx
```
