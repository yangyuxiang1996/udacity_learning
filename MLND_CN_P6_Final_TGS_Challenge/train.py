from data_processing.dataloader import CreateDataset
from model.Models import UnetSeNext50, UNetResNet34
from contextlib import contextmanager
from tensorboardX import SummaryWriter
import time 
import os
import sys
sys.path.append('../')


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# define some parameters
TRAIN_PATH = './data/train/'
loss_type = 'lovasz'
optimizer = 'SGD'
# lr_scheduler = 'CosineAnnealingLR'
lr_scheduler = 'cycle'
pretrained = True
n_epoch = 130 
batch_size = 32
Net = UnetSeNext50
learning_rate = 0.001
T_MAX = 60    
T_MUL = 1
LR_MIN = 0

writer = SummaryWriter(log_dir='./log3')

train_dataset = CreateDataset(TRAIN_PATH)
loaders, ids = train_dataset.yield_dataloader(num_workers=11, batch_size=batch_size, nfold=5)

for i, (train_loader, val_loader) in enumerate(loaders, 1):
    with timer('Fold {}'.format(i)):
        net = Net(lr=learning_rate, fold=i)
        net.define_metric(loss_type)
        net.define_optmizer_schedular(mode=lr_scheduler)
        net.train_and_valid(train_loader, val_loader, writer, epoch=n_epoch)

writer.close()




