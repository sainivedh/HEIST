from yolov5.inference_torch_hub import yolo_inference
from config import YOLO_WEIGHTS_PATH, TEST_IMG_FOLDER_PATH, \
    Cropped_YOLO_FOLDER_PATH, ctc_weights_path, seq2seq_weights_path
from cropped_data.inference import ctc_inference
from Transliteration.model import test, Transliteration_EncoderDecoder_Attention
from TexttoSpeech.main import texttospeech

import glob
import cv2
import os

img_path_crop = glob.glob(Cropped_YOLO_FOLDER_PATH+'/*.jpg')
for file in img_path_crop:
    os.remove(file)

#texttospeech('Happy International Womens day to all', 'en')
img_path = os.path.join(TEST_IMG_FOLDER_PATH, '4.jpg')
boxes = yolo_inference(img_path, YOLO_WEIGHTS_PATH, Cropped_YOLO_FOLDER_PATH, key=1)
inp_img = cv2.imread(img_path)

for bbox in boxes:
    xl, yl, xr, yr = bbox
    cv2.rectangle(inp_img, (xl, yl), (xr, yr), (0, 255, 0), 2)
cv2.imwrite('bhopal_yolo.jpg',inp_img)
cv2.imshow('Hindi Text Detected', inp_img)
cv2.waitKey(0)

print('Cropping Done')
#img_path_crop = os.path.join(Cropped_YOLO_FOLDER_PATH, 'crop_1.jpg')
img_path_crop = glob.glob(Cropped_YOLO_FOLDER_PATH+'/*.jpg')
#print(img_path_crop)
all_eng_strings = []
for img_path in img_path_crop:
    decode_strings = ctc_inference(img_path, ctc_weights_path, 3)
    print('Decoding Done')
    eng_strings = []
    for hin_text in decode_strings:
        eng_strings.append(test(seq2seq_weights_path, hin_text))
    all_eng_strings.append(eng_strings)
#print(eng_strings)

for index, bbox in enumerate(boxes):
    xl, yl, xr, yr = bbox
    cv2.rectangle(inp_img, (xl, yl), (xr, yr), (75, 0, 130), -1)
    for i in range(3):
        cv2.putText(inp_img, f'{all_eng_strings[index][i]}', (xl+5, yl+35*(i+1)),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0),2,cv2.LINE_AA)
cv2.imwrite('bhopal.jpg',inp_img)
cv2.imshow('Detected Text', inp_img)
cv2.waitKey(0)
#print(decode_strings)
#texttospeech(decode_strings[0])

'''
path = os.path.join(TEST_IMG_FOLDER_PATH, '2.jpg')
img = cv2.imread(path)
cv2.imshow('out', img)
cv2.waitKey(0)
'''