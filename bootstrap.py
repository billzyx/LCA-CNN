from __future__ import absolute_import
from __future__ import print_function

import os
import shutil
import sys
import tarfile
from shutil import copyfile

import matplotlib.pyplot as plt

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    # Not Python 3 - today, it is most likely to be Python 2
    # But note that this might need an update when Python 4
    # might be around one day
    from urllib import urlretrieve


def download_file(url, dest=None):
    if not dest:
        dest = os.path.join('.', url.split('/')[-1])
    urlretrieve(url, dest)


train_image_path = 'CUB_200_2011.tgz'
if not os.path.isfile(train_image_path):
    print('Downloading images...')
    download_file('http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz')
tarfile.open(train_image_path).extractall(path='.')

# Change the path to your dataset folder:
base_folder = 'CUB_200_2011/'

# These path should be fine
images_txt_path = base_folder + 'images.txt'
train_test_split_path = base_folder + 'train_test_split.txt'
images_path = base_folder + 'images/'

# Here declare where you want to place the train/test folders
# You don't need to create them!
test_folder = 'test/'
train_folder = 'train/'

print('Splitting images...')


def ignore_files(dir, files): return [f for f in files if os.path.isfile(os.path.join(dir, f))]


shutil.copytree(images_path, test_folder, ignore=ignore_files)
shutil.copytree(images_path, train_folder, ignore=ignore_files)

with open(images_txt_path) as f:
    images_lines = f.readlines()

with open(train_test_split_path) as f:
    split_lines = f.readlines()

test_images, train_images = 0, 0

for image_line, split_line in zip(images_lines, split_lines):

    image_line = (image_line.strip()).split(' ')
    split_line = (split_line.strip()).split(' ')

    image = plt.imread(images_path + image_line[1])

    # If test image
    if (int(split_line[1]) is 0):
        copyfile(images_path + image_line[1], test_folder + image_line[1])
        test_images += 1
    else:
        if len(image.shape) == 3:
            # If train image
            copyfile(images_path + image_line[1], train_folder + image_line[1])
            train_images += 1

print(train_images, test_images)
assert train_images == 5990
assert test_images == 5794

print('Dataset succesfully splitted!')
