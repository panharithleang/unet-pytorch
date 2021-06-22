import random
import glob
import shutil

sample_index = random.sample(range(1311), k=1311)
training = sample_index[:838]
validation = sample_index[838: 1048]
test = sample_index[1048:]

for idx, sample in enumerate(training):
    img_src_path = f'/Users/leangpanharith/Documents/school_stuffs/unet/cleaned_dataset/{sample}.jpg'
    mask_src_path = f'/Users/leangpanharith/Documents/school_stuffs/unet/cleaned_dataset/{sample}_mask.jpg'
    img_dst_path = f'/Users/leangpanharith/Documents/school_stuffs/unet/data/training/{idx}.jpg'
    mask_dst_path = f'/Users/leangpanharith/Documents/school_stuffs/unet/data/training/{idx}_mask.jpg'
    shutil.copy(img_src_path, img_dst_path)
    shutil.copy(mask_src_path, mask_dst_path)

for idx, sample in enumerate(validation):
    img_src_path = f'/Users/leangpanharith/Documents/school_stuffs/unet/cleaned_dataset/{sample}.jpg'
    mask_src_path = f'/Users/leangpanharith/Documents/school_stuffs/unet/cleaned_dataset/{sample}_mask.jpg'
    img_dst_path = f'/Users/leangpanharith/Documents/school_stuffs/unet/data/validation/{idx}.jpg'
    mask_dst_path = f'/Users/leangpanharith/Documents/school_stuffs/unet/data/validation/{idx}_mask.jpg'
    shutil.copy(img_src_path, img_dst_path)
    shutil.copy(mask_src_path, mask_dst_path)

for idx, sample in enumerate(test):
    img_src_path = f'/Users/leangpanharith/Documents/school_stuffs/unet/cleaned_dataset/{sample}.jpg'
    mask_src_path = f'/Users/leangpanharith/Documents/school_stuffs/unet/cleaned_dataset/{sample}_mask.jpg'
    img_dst_path = f'/Users/leangpanharith/Documents/school_stuffs/unet/data/test/{idx}.jpg'
    mask_dst_path = f'/Users/leangpanharith/Documents/school_stuffs/unet/data/test/{idx}_mask.jpg'
    shutil.copy(img_src_path, img_dst_path)
    shutil.copy(mask_src_path, mask_dst_path)

# print(len(sample_index))
