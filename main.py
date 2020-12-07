import torchvision as tv
import numpy as np
from torch.utils.data import DataLoader
from tqdm import trange
from functions import to_onehot, get_item
from Net import Net


transform = tv.transforms.Compose([
    tv.transforms.RandomRotation(degrees=10),
    tv.transforms.ToTensor(),
])
trainDataset = tv.datasets.MNIST(
    root="./data", # 下载数据，并且存放在data文件夹中
    train=True,
    transform=transform,
    download=True
)
trainLoader = DataLoader(dataset=trainDataset, shuffle=True)
testDataset = tv.datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)
testLoader = DataLoader(dataset=testDataset, shuffle=True)


net = Net(lr=1e-3)
i = 0
avg_loss = 0
epoch = 20
# with trange(epoch * len(trainLoader)) as t:
#     for ep in range(1, epoch+1):
#         for x, y in trainLoader:
#             x = x.numpy().reshape((28*28, 1))
#             y = to_onehot(y.item())
#
#
#
#             y_pred = net.forward(x)
#             loss = get_item(net.backward(y))
#             avg_loss = (avg_loss * i + loss) / (i+1)
#             net.step()
#
#             t.set_postfix(loss="%.6f" % loss, avg_loss = "%.6f" % avg_loss)
#
#
#             t.update(1)
#             i += 1
#             if i == 100:
#                 i = 0
#                 avg_loss = 0
# net.save()

corr = 0
net.load()

with trange(len(testLoader)) as t:
    for x, y in testLoader:
        x = x.numpy().reshape((28 * 28, 1))
        y = y.item()
        yy = to_onehot(y)


        y_pred = net.forward(x)
        loss = get_item(net.backward(yy))
        if y == np.argmax(y_pred):
            corr+=1

        t.set_postfix(loss = loss)
        t.update(1)
print(corr / len(testLoader) * 100, ' %%')