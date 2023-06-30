from torch.utils.tensorboard import SummaryWriter
import json

# 加载json数据
with open('./utils/cls/loss.json', 'r') as f:
    loss_data = json.load(f)

with open('./utils/cls/top1.json', 'r') as f:
    top1_data = json.load(f)

with open('./utils/cls/top5.json', 'r') as f:
    top5_data = json.load(f)

with open('./utils/cls/val_top1.json', 'r') as f:
    val_top1_data = json.load(f)

# 创建TensorBoard的SummaryWriter
writer = SummaryWriter("utils/runs_cls")

# 将数据写入TensorBoard日志文件
for step, loss_value in enumerate(loss_data):
    writer.add_scalar('Loss', loss_value[2], step)

for step, top1_value in enumerate(top1_data):
    writer.add_scalar('Top1 Accuracy', top1_value[2], step)

for step, top5_value in enumerate(top5_data):
    writer.add_scalar('Top5 Accuracy', top5_value[2], step)

for step, val_top1_data in enumerate(val_top1_data):
    writer.add_scalar('Val Top1 Accuracy', val_top1_data[2], step)

# 关闭SummaryWriter
writer.close()