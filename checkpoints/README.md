# Checkpoints

不要将大权重文件提交到仓库。

将下载得到的模型权重放在本目录下，或在命令行中通过 `--weights` 指向任意外部路径。

建议文件命名：

- `checkpoints/DAUB-R.pt`
- `checkpoints/ITSDT-15k.pt`
- `checkpoints/IRDST-H.pt`

验证示例：

```bash
python val.py --weights checkpoints/DAUB-R.pt --data data.yaml --dataset-root /path/to/DAUB-R_retina --device 0
```
