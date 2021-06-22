import glob
import cv2
import os
import json
import numpy as np

for idx, file in enumerate(glob.glob('/Users/leangpanharith/Documents/school_stuffs/unet/dataset/pic+json+cv2_mask/json/*.json')):
    with open(file, 'r') as json_file:
        file_info = json.load(json_file)
        data_src = "/Users/leangpanharith/Documents/school_stuffs/unet/cleaned_dataset"
        file_name = os.path.basename(file).split('.')[0]
        img_path = f"/Users/leangpanharith/Documents/school_stuffs/unet/dataset/pic+json+cv2_mask/pic/{file_name}.png"
        img = cv2.imread(img_path)
        mask = np.zeros(img.shape[:2])
        mask_fill = file_info['shapes']
        for fill in mask_fill:
            if fill['label'] == 'right':
                points = np.array(fill['points'], dtype=int)
                
                cv2.drawContours(mask, [points], 0 ,255, -1)
                cv2.imwrite(f'{data_src}/{idx}.jpg', img)
                cv2.imwrite(f'{data_src}/{idx}_mask.jpg', mask)
