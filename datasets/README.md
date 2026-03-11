# Datasets

不要将真实数据集文件提交到仓库。

推荐将 retina 处理后的数据集按如下结构放置到本目录下，或者通过 `--dataset-root` 指向任意外部目录：

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

- `images/` 为主图像目录。
- `image/` 为与 `images/` 一一对应的第二模态目录。
- 代码会自动将 `images/.../xxx.png` 配对到 `image/.../xxx.png`，并在读取时拼成 6 通道输入。
- 标签采用 YOLO 检测格式，放在 `labels/train` 与 `labels/test` 中。

若使用 `ITSDT-15K` 或 `IRDST-H`，目录结构保持一致即可，只需将根目录名替换为对应数据集名称。
