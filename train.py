import functools
import time

import numpy as np
from torchvision.datasets.folder import ImageFolder
import torchvision.transforms as trans
import torch.cuda as cuda
import matplotlib.pyplot as plt
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from model import WideResnet

# --------------------------------Hyperparameters--------------------------#
BATCH_SIZE = 10
EPOCH = 50
LEARNING_RATE = 1e-3


# --------------------------------End Hyperparameters----------------------#
def split_indices(n, val_pct):
    indices = np.random.permutation(n)
    bound = int(n * val_pct)
    return indices[bound:], indices[:bound]


def get_default_device():
    if cuda.is_available():
        return torch.device('cuda')

    return torch.device("cpu")


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# wrapped dataloader
class DeviceDataloader:

    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        for batch_tuple in self.dataloader:
            yield to_device(batch_tuple, self.device)

    def __len__(self):
        return len(self.dataloader)


BASE_DIR = "./data"

transform = trans.Compose([trans.RandomCrop(256, padding=20, padding_mode='reflect'),
                           trans.RandomHorizontalFlip(),
                           trans.ToTensor()])  # dont know how to calculate mean and std before images are tensorized.
dataset = ImageFolder(BASE_DIR + "/train", transform=transform)

# print(dataset)
train_idx, val_idx = split_indices(len(dataset), 0.1)

print("length of train idx: " + str(len(train_idx)))
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

train_dl = DeviceDataloader(DataLoader(dataset, BATCH_SIZE, sampler=train_sampler,
                                       pin_memory=True), get_default_device())
val_dl = DeviceDataloader(DataLoader(dataset, BATCH_SIZE, sampler=val_sampler,
                                     pin_memory=True), get_default_device())

model = WideResnet()
to_device(model, get_default_device())

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# optimizer = torch.optim.Adadelta(model.parameters(), lr=LEARNING_RATE, rho=0.9, eps=1e-6, weight_decay=0)
avg_validation_losses = []
avg_validation_accuracy = []

print("length of training dataloader: " + str(len(train_dl)))
start_time = time.time()
for i in range(EPOCH):
    print("EPOCH: " + str(i))
    count = 0
    for data_batch, target_batch in train_dl:
        if count % 10 == 0:
            print("batch: " + str(count))
        count += 1
        cp = model.train_step(data_batch, target_batch)
        cp.backward()
        optimizer.step()
        optimizer.zero_grad()
        # torch.cuda.empty_cache()
    if i % 1 == 0 or i == EPOCH - 1:
        obj = []
        with torch.no_grad():
            for data_batch, target_batch in val_dl:
                obj.append(model.validation_step(data_batch, target_batch))
        losses = [ele["loss"] for ele in obj]
        avg_loss = sum(losses) / len(losses)
        accs = [ele["accuracy"] for ele in obj]
        avg_accs = sum(accs) / len(accs)
        avg_validation_losses.append(avg_loss)
        avg_validation_accuracy.append(avg_accs)
        print("validation loss: " + str(avg_loss))
        print("validation accuracy: " + str(avg_accs))

end_time = time.time()

print("Training time: "  + str(end_time - start_time))
# filename = filetype_batchsize_epoches_optalgorithm_optparams.pt
torch.save(model.state_dict(), "./sdict_10_50_adm_1e-3.pt")

timefile = open("training_time.txt", "w+")
timefile.write(str(end_time - start_time))

# data visualization
x = [1 * i for i in range(EPOCH)]

plt.plot(x, avg_validation_losses, label="validation loss")
plt.xlabel("epoches")
plt.ylabel("loss")
plt.show()

plt.plot(x, avg_validation_accuracy, label="validation accuracy")
plt.xlabel("epoches")
plt.ylabel("accuracy")
plt.show()

file_loss = open("loss_3.txt", "w+")
file_loss.writelines([str(ele) + "\n" for ele in avg_validation_losses])
file_accr = open("accuracy_3.txt", "w+")
file_accr.writelines([str(ele) + "\n" for ele in avg_validation_accuracy])

