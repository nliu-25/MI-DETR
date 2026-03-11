# Dataset And Weight Downloads

## 1. 原始数据集

原始数据集下载入口请参考 MoPKL 项目：

- DAUB-R / ITSDT-15K / IRDST-H: <https://github.com/UESTC-nnLab/MoPKL>

## 2. retina 处理后数据集

论文复现推荐直接使用 retina 处理后的数据集。

- 文件名：`Dataset_retina`
- 链接：<https://pan.baidu.com/s/1p5409A7rldXrFzzcwC_ALQ?pwd=5paw>
- 提取码：`5paw`

建议解压后按数据集分别整理，例如：

- `datasets/DAUB-R_retina`
- `datasets/ITSDT-15K_retina`
- `datasets/IRDST-H_retina`

## 3. 训练权重

训练权重与 retina 数据集使用同一百度网盘入口获取。下载后请手动放置到 `checkpoints/` 目录，或在运行命令时通过 `--weights` 指定路径。

建议命名：

- `checkpoints/DAUB-R.pt`
- `checkpoints/ITSDT-15k.pt`
- `checkpoints/IRDST-H.pt`

## 4. 开源仓库中不包含的内容

以下内容不会上传到 GitHub：

- 原始数据集文件本体
- retina 处理后数据集文件本体
- 大模型权重文件本体
- 本地训练日志与缓存
