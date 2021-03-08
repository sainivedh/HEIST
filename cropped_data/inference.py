import torch

import config
import dataset_new
import engine_new
from model_practice import CaptchaModel

from practice import beam_search_decoder

import data
from torch.autograd import Variable
from PIL import Image
from train_new import decode_predictions, remove_blanks

'''
cm = torch.load('model_57000.pt')
cm.to(config.DEVICE)
transformer = data.resizeNormalize((100, 32))
image = Image.open('643.jpg').convert('L')
#print(image.size)
image = transformer(image)
#print(image.shape)
    #if torch.cuda.is_available():
    #    image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)
image = image.to(config.DEVICE)
cm.eval()
#print(image.shape)
_, preds = cm(image)
#print(preds.shape)
'''

hindi_alphabets = [chr(alpha) for alpha in range(2304, 2432)]
hindi_alphabet_size = len(hindi_alphabets)
hindi_alpha2index = {}
hindi_index2alpha = {}
for index, alpha in enumerate(hindi_alphabets):
    hindi_alpha2index[alpha] = index + 1
hindi_index2alpha = {value:key for key,value in hindi_alpha2index.items()}
hindi_text = "वारंगल"
op = []
for char in hindi_text:
    op.append(hindi_alpha2index[char])
op = torch.tensor(op)

'''
best_k_preds = beam_search_decoder(preds, 3)
current_k_preds = []
for i in range(2, -1, -1):
    true_preds, current_preds = decode_predictions(op, best_k_preds[i][0], hindi_index2alpha)
    current_k_preds.append(current_preds)


#print(remove_blanks(current_k_preds[2][0]))
#print(current_preds)
for ls in current_k_preds:
    print(remove_blanks(ls).replace('र््','र्'))
'''

def ctc_inference(img_path=None, ctc_weights=None, top_k = 3):
    cm = torch.load(ctc_weights)
    cm.to(config.DEVICE)
    transformer = data.resizeNormalize((100, 32))
    image = Image.open(img_path).convert('L')
    # print(image.size)
    image = transformer(image)
    # print(image.shape)
    # if torch.cuda.is_available():
    #    image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)
    image = image.to(config.DEVICE)
    cm.eval()
    # print(image.shape)
    _, preds = cm(image)
    # print(preds.shape)

    best_k_preds = beam_search_decoder(preds, top_k)
    current_k_preds = []
    for i in range(top_k-1, -1, -1):
        true_preds, current_preds = decode_predictions(op, best_k_preds[i][0], hindi_index2alpha)
        current_k_preds.append(current_preds)

    # print(remove_blanks(current_k_preds[2][0]))
    # print(current_preds)
    out_strings = []
    for ls in current_k_preds:
        out_strings.append(remove_blanks(ls).replace('र््', 'र्'))

    return out_strings
