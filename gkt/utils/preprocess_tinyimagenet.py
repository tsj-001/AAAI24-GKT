import os
import shutil


dataset_base = '/remote-home/share/tinyimagenet/val'


with open(os.path.join(dataset_base, 'val_annotations.txt'), 'r') as f:
    annotations = f.readlines()

for anno in annotations:
    img_name, label = anno.split('\t')[:2]
    img_path = os.path.join(os.path.join(dataset_base, 'images'), img_name)
    target_folder_path = os.path.join(dataset_base, label)

    if not os.path.exists(target_folder_path):
        os.makedirs(target_folder_path)

    shutil.copyfile(img_path, os.path.join(target_folder_path, img_name))

shutil.rmtree(os.path.join(dataset_base, 'images'))

print('Done!')
