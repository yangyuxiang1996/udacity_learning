# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import gc
import os
import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from model.Models import UnetSeNext50, UNetResNet34
from data_processing.dataloader import CreateDataset, do_horizontal_flip
from loss.Evaluation import do_length_decode, do_length_encode


#%%
def predict(nets, test_path, load_paths, batch_size=32, tta_transform=None, threshold=0.5, min_size=0):
    prob_sum = np.zeros((18000, 1, 101, 101))
    test_dataset = CreateDataset(test_path)
    test_loader, _ = test_dataset.yield_dataloader(data='test', num_workers=11, batch_size=batch_size)
    # predict
    for i in tqdm(range(len(load_paths))):
        if 'UNetResNet34' in load_paths[i]:
            net = nets[0]
        elif 'UnetSeNext50' in load_paths[i]:
            net = nets[1]
        
        net.load_model(load_paths[i])
        p = net.do_predict(test_loader, threshold=0, tta_transform=tta_transform)
        prob_sum += p['pred']
        del net
    del test_dataset, test_loader
    
    return prob_sum, p


#%%
def tta_transform(images, mode):
    out = []
    if mode == 'out':
        images = images[0]
    images = images.transpose((0, 2, 3, 1))
    tta = []
    for i in range(len(images)):
        t = np.fliplr(images[i])
        tta.append(t)
    tta = np.transpose(tta, (0, 3, 1, 2))
    out.append(tta)
    return np.asarray(out)


#%%
TEST_PATH = '/home/18310206637/project/MLND_CN_P6_Final_TGS_Challenge/data/test/'
net = [UNetResNet34(), UnetSeNext50()]
# NET1_NAME = type(net1).__name__
# NET2_NAME = type(net2).__name__
THRESHOLD = 0.5
MIN_SIZE = 0
BATCH_SIZE = 128


#%%
filelists = [ 'models/UnetSeNext50/2019-08-15 14:17_Fold2_Epoach116_val0.877', 
                'models/UnetSeNext50/2019-08-15 21:00_Fold3_Epoach123_val0.870']
prob_sum, p = predict(net, TEST_PATH, filelists,
                    tta_transform=tta_transform,
                    batch_size=BATCH_SIZE,
                    threshold=THRESHOLD,min_size=MIN_SIZE)


#%%
avg = prob_sum / (len(filelists))
pred = avg > THRESHOLD


#%%
rle = []
for i in range(len(pred)):
    rle.append(do_length_encode(pred[i]))
# create sub
df = pd.DataFrame(dict(id=p['id'], rle_mask=rle))


#%%
# You can modify the filename of the .csv file
df.to_csv(os.path.join(
        './results/',
        'UnetSeNext50.csv'),
        index=False)


#%%



