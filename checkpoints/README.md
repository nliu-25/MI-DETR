# Checkpoints

不要将大权重文件提交到仓库。

将下载得到的模型权重放在本目录下，或在命令行中通过 `--weights` 指向任意外部路径。

现有权重属于已发布的下载内容。本次 TIP 文档和术语更新没有产生新权重，也不意味着这些权重已与修订稿中的全部实验绑定。复现时请记录 checkpoint 的 SHA-256，并核对模型配置、数据划分与评估协议。

建议文件命名：

- `checkpoints/DAUB-R.pt`
- `checkpoints/ITSDT-15k.pt`
- `checkpoints/IRDST-H.pt`

验证示例：

```bash
python val.py --weights checkpoints/DAUB-R.pt --data data.yaml --dataset-root /path/to/DAUB-R_retina --device 0
```
