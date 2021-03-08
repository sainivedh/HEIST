from tqdm import tqdm
import torch
import config
from torch.nn import functional as F
from torch import nn
import numpy as np

def train_fn(model, dataset, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    fin_loss = 0
    images_batch, targets_batch = dataset.get_batch(config.BATCH_SIZE)

    for i in range(config.BATCH_SIZE):
        image, target = images_batch[i].to(config.DEVICE), targets_batch[i].to(config.DEVICE)
        bs, output = model(image)
        log_probabs = F.log_softmax(output, 2)
        #print(log_probabs.size(0))
        target_length = torch.tensor(len(target), dtype=torch.int32)
        input_length = torch.tensor(np.full((bs,), log_probabs.size(0), dtype=np.int32))
        loss = criterion(log_probabs, target, input_length, target_length)
        loss /= config.BATCH_SIZE
        loss.backward(retain_graph=True)
        fin_loss += loss

    optimizer.step()
    return fin_loss/config.BATCH_SIZE

def eval_fn(model, dataset, criterion):
    model.eval()
    fin_loss = 0
    fin_preds = []
    orig_preds = []
    #tk = tqdm(data_loader, total=len(data_loader))
    images_batch, targets_batch = dataset.get_batch(config.BATCH_SIZE)
    with torch.no_grad():
        for i in range(config.BATCH_SIZE):
            image, target = images_batch[i].to(config.DEVICE), targets_batch[i].to(config.DEVICE)
            bs, output = model(image)
            log_probabs = F.log_softmax(output, 2)
            target_length = torch.tensor(len(target), dtype=torch.int32)
            input_length = torch.tensor(np.full((bs,), log_probabs.size(0), dtype=np.int32))
            loss = criterion(log_probabs, target, input_length, target_length)
            loss /= config.BATCH_SIZE
            fin_loss += loss
            fin_preds.append(output)
            orig_preds.append(target.cpu())
        return orig_preds, fin_preds, fin_loss/config.BATCH_SIZE