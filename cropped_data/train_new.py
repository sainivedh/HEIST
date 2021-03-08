import os
import glob
import torch
import numpy as np
from tqdm import tqdm

from sklearn import model_selection
from sklearn import metrics

import config
import dataset_new
import engine_new
from model_practice import CaptchaModel

from torch import nn

def remove_blanks(preds):
    out = []
    i = 0
    #print(len(preds))
    while (i < len(preds)):
        out.append(preds[i])
        temp = preds[i]
        #print(temp)
        i = i + 1
        while (i < len(preds) and preds[i] == temp):
            i = i + 1
            #print(i)

    temp = "".join(out)
    return temp.replace('§','')


def decode_predictions(orig_preds, preds, encoder):
    #preds = preds.permute(1,0,2) #b,w,val
    #preds = torch.softmax(preds, 2)
    #preds = torch.argmax(preds, 2)
    #preds = preds.detach().cpu().numpy()
    cap_preds = []
    true_preds = []
    #print(preds)

    #for j in range(preds.shape[0]):
    temp = []
    for k in preds:
            #k = k - 1
        if k == 0:
            temp.append("§")
                #temp.append(0)
        else:
            temp.append(encoder[k])
                #temp.append(k)
    tp = "".join(temp)
    cap_preds.append(tp)

    temp = []
    #print(orig_preds)
    for j in orig_preds.numpy():
        temp.append(encoder[j])
    tp = "".join(temp)
    true_preds.append(tp)
    #print(cap_preds)
    return true_preds, cap_preds


'''

def decode_predictions(orig_preds, preds, encoder):
    preds = preds.permute(1,0,2) #b,w,val
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    cap_preds = []
    true_preds = []
    #print(preds)

    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            #k = k - 1
            if k == 0:
                temp.append("§")
                #temp.append(0)
            else:
                temp.append(encoder[k])
                #temp.append(k)
        tp = "".join(temp)
        cap_preds.append(tp)

    temp = []
    #print(orig_preds)
    for j in orig_preds.numpy():
        temp.append(encoder[j])
    tp = "".join(temp)
    true_preds.append(tp)
    #print(cap_preds)
    return true_preds, cap_preds

'''

def run_training():

    file = open('annotations.txt', encoding="utf-8").readlines()
    annots = []
    images_path = []
    for line in file:
        annots.append(line.split()[1].strip())
        images_path.append(line.split()[0].strip())

    hindi_alphabets = [chr(alpha) for alpha in range(2304, 2432)]
    hindi_alphabet_size = len(hindi_alphabets)
    hindi_alpha2index = {}
    hindi_index2alpha = {}
    for index, alpha in enumerate(hindi_alphabets):
        hindi_alpha2index[alpha] = index + 1
    hindi_index2alpha = {value:key for key,value in hindi_alpha2index.items()}
    #print(hindi_index2alpha[6])
    outputs = []
    for out in annots:
        l = []
        for c in out:
            l.append(hindi_alpha2index[c])
        outputs.append(l)

    (
        train_imgs,
        test_imgs,
        train_targets,
        test_targets,
        _,
        test_targets_orig,
    ) = model_selection.train_test_split(
        images_path, outputs, annots, test_size=0.05, random_state=550, shuffle=True)

#images_path[100:len(images_path)], images_path[0:100], outputs[100:len(outputs)], outputs[0:100], annots[100:len(annots)], annots[0:100]


    train_dataset = dataset_new.ClassificationDataset(
        image_paths=images_path,
        targets=outputs,
    )

    test_dataset = dataset_new.ClassificationDataset(
        image_paths=test_imgs,
        targets=test_targets,
    )

    """
        images_batch, targets_batch = train_dataset.get_batch(8)
        for img,tar in zip(images_batch, targets_batch):
            print(img.shape)
            print(tar)
            print('-------------------')
        return
    """
    #model = CaptchaModel(32,1,hindi_alphabet_size+1,256)
    model = torch.load('model_yolonewlr2500.pt')
    model.to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )
    criterion = nn.CTCLoss(blank=0)
    for n_batch in range(config.EPOCHS):
        epoch = int((n_batch * 8) / len(images_path))
        train_loss = engine_new.train_fn(model, train_dataset, optimizer, criterion)
        orig_preds, valid_preds, test_loss = engine_new.eval_fn(model, test_dataset, criterion)
        valid_captcha_preds = []
        true_captcha_preds = []
        for op, vp in zip(orig_preds, valid_preds):
            true_preds, current_preds = decode_predictions(op, vp, hindi_index2alpha)
            valid_captcha_preds.extend(current_preds)
            true_captcha_preds.extend(true_preds)
        combined = list(zip(true_captcha_preds, valid_captcha_preds))
        print(combined[:4])
        #test_dup_rem = []
        print(
            f"N_Batches={n_batch} Epoch={epoch}"
        )
        if (n_batch % 500 == 0):
            torch.save(model, f'model_yolonewlr{n_batch+2500}.pt')
        #scheduler.step(test_loss)



if __name__ == "__main__":
    run_training()
