# Datasets

不要将真实数据集文件提交到仓库。

将表观图像与预生成运动图按如下结构放置到本目录下，或者通过 `--dataset-root` 指向任意外部目录。`_retina` 是已有下载内容沿用的名称；修订稿中的运动建模模块称为 RIMCA。

```text
datasets/
└── DAUB-R_retina/
    ├── images/
    │   ├── train/
    │   └── test/
    ├── image/
    │   ├── train/
    │   └── test/
    └── labels/
        ├── train/
        └── test/
```

说明：

- `images/` 为表观图像目录，即原始红外图像。
- `image/` 为与 `images/` 一一对应的运动图像目录。
- 代码会自动将 `images/.../xxx.png` 配对到 `image/.../xxx.png`，并在读取时拼成 6 通道输入。
- 标签采用 YOLO 检测格式，放在 `labels/train` 与 `labels/test` 中。
- 以上为已有数据包的目录示例。默认 `data.yaml` 将 `images/test` 映射到 `val`；开展新实验时，应在独立 YAML 中配置真实的 train/val/test 划分，使用独立 val 选择 checkpoint，冻结后再评估 test。
- 当前训练和验证入口不在线生成运动图。运动图的帧序、序列重置边界和生成版本需与目标实验一致。

若使用 `ITSDT-15K` 或 `IRDST-H`，目录结构保持一致即可，只需将根目录名替换为对应数据集名称。
