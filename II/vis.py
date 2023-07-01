import json
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("./logs/vit11")
DATA_PATH = "./log/vit11/vit11_9790e_00000/result.json"
data = []
for line in open(DATA_PATH, "r"):
    data.append(json.loads(line))

keys = [
    "train_loss", "dev_loss", "dev_top1_accuracy", "dev_top5_accuracy"
]

for i, dic in enumerate(data):
    for key in keys:
        writer.add_scalar(key, dic[key], i + 1)
writer.close()
