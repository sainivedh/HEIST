import torch
from dataset import eng_alphabets, hindi_alpha2index, eng_alpha2index
from dataset import infer
from model import net_attn





def test(net, word, device = 'cpu'):
    net = net.eval().to(device)
    outputs = infer(net, word, 30)
    eng_output = ''
    for out in outputs:
        val, indices = out.topk(1)
        index = indices.tolist()[0][0]
        if index == 0:
            break
        eng_char = eng_alphabets[index-1]
        eng_output += eng_char
    print(word + ' - ' + eng_output)
    return eng_output

print(test(net_attn, 'ज्योति'))
