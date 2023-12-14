# MAPA2023-MoveNet

本仓库为北京航空航天大学2023秋通识课程MAPA期末作业，依照*Apache 2.0*协议部分引用了[MoveNet Demo](https://github.com/tensorflow/docs/blob/master/site/en/hub/tutorials/movenet.ipynb)代码。

![Still Shape Form](https://github.com/NightingaleCen/MAPA2023-MoveNet/blob/main/Still%20Shape%20Form.png)

## 环境配置

```shell
conda create -n movenet python=3.9.18
```

```shell
conda activate movenet
```

```shell
pip install -r requirements.txt
```

## 运行Demo

请确保设备安装有可以运行的摄像头。

```
python camera.py
```

| 按键 | 功能                 |
| ---- | -------------------- |
| q    | 退出                 |
| c    | 切换摄像头           |
| o    | 显示关键点信息       |
| ]    | 提高关键点置信度阈值 |
| [    | 降低关键点置信度阈值 |
| p    | 显示当前动作分类     |

## 训练分类器

[`pose_images/keypoints.json`](pose_images/keypoints.json)中包含超过两千张人体姿势图片的关键点和标签信息，训练得到的姿势分类器权重将被保存在`pose_classifier.h5`中。

```shell
python pose_classifier.py
```

## LICENSE

[Apache License 2.0](https://opensource.org/licenses/Apache-2.0)
