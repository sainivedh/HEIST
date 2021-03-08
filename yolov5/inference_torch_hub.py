import cv2
import os
import torch
import numpy as np
import numpy

'''
model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model='best-250epochs.pt')
img_name = "Mumbai"
img = cv2.imread(f'{img_name}.jfif')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgs = [img]

model.conf = 0.4
model.iou = 0.2

results = model(imgs, size=416)

results.show()
print(len(results.xyxy[0]))


for i in range(len(results.xyxy[0])):
    bbox = list(results.xyxy[0][i].numpy())[:4]
    xl, yl, xr, yr = list(map(int, bbox))
    img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_crop = img_BGR[yl:yr, xl:xr]
    cv2.imwrite(f'{img_name}_crop_{i+1}.jpg', img_crop)

'''

def yolo_inference(img_path=None, weights_path=None, out_crop_path=None, key=1):

    model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=weights_path)
    #img_name = "Mumbai"

    if key == 1:

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs = [img]

        model.conf = 0.4
        model.iou = 0.2

        results = model(imgs, size=416)

        boxes = []

        #results.show()
        #print(len(results.xyxy[0]))


        '''
        bbox = list(results.xyxy[0][0].numpy())[:4]
        xl, yl, xr, yr = list(map(int, bbox))
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img[yl:yr, xl:xr]
        cv2.imwrite('bhopal_crop.jpg', img)
        '''


        for i in range(len(results.xyxy[0])):
            bbox = list(results.xyxy[0][i].numpy())[:4]
            xl, yl, xr, yr = list(map(int, bbox))
            boxes.append([xl, yl, xr, yr])

            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_crop = img_BGR[yl:yr, xl:xr]
            #cv2.imwrite(f'../Cropped_YOLO_images/{img_num}_crop_{i + 1}.jpg', img_crop)
            #os.path.join(out_path, f'{img_num}_crop_{i + 1}.jpg')
            cv2.imwrite(os.path.join(out_crop_path, f'crop_{i + 1}.jpg'), img_crop)


    else:

        img = cv2.imread(img_path)
        boxes = []
        h,w,_ = img.shape
        boxes.append([0,0,w-10,h-10])
        cv2.imwrite(os.path.join(out_crop_path, f'crop_.jpg'), img)

    return boxes



'''
bbox = list(results.xyxy[0][0].numpy())[:4]
xl, yl, xr, yr = list(map(int, bbox))
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
img = img[yl:yr, xl:xr]
cv2.imwrite('bhopal_crop.jpg', img)
'''
