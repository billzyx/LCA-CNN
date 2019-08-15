from __future__ import print_function

__author__ = 'tianwai'

from PIL import Image
import os
import sys
from multiprocessing.dummy import Pool as ThreadPool

input_dir = os.getcwd() + ""  # path to img source folder
train_dir = os.getcwd() + "/train"  # train directory
train_dir2 = os.getcwd() + "/trainRotate"  # train directory


def readf():
    arr = []
    print("starting....")
    print("Colecting data from %s " % train_dir)
    tclass = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    counter = 0
    if not os.path.exists(train_dir2):
        os.makedirs(train_dir2)
    for x in tclass:
        list_dir = os.path.join(train_dir, x)
        list_tuj = os.path.join(train_dir2, x)

        if not os.path.exists(list_tuj):
            os.makedirs(list_tuj)

        for d in os.listdir(list_dir):
            try:
                if not os.path.isfile(os.path.join(list_dir, d)):
                    continue
                if str(d).startswith('.'):
                    continue
                arr += [(d, train_dir, x)]
            except Exception as e:
                print("Error resize file : %s - %s " % (x, d))
                sys.exit(1)
        counter += 1

    pool = ThreadPool(16)
    results = pool.map(process_single_image, arr)
    pool.close()
    pool.join()
    print('Total image:')
    print(len(set(sum(results, []))))


def process_single_image(tr):
    d, dir_name, x = tr
    img_o = Image.open(os.path.join(dir_name, x, d)).convert('RGB')
    fname, extension = os.path.splitext(d)
    direction = 0 - 15
    cnt_arr = []
    while direction < 360:
        direction += 15
        if direction >= 90 and direction <= 270:
            continue
        img = img_o.rotate(direction)
        new_file = fname + '_' + str(direction) + '.jpg'
        new_file_path = os.path.join(train_dir2, x, new_file)
        img.save(new_file_path, "JPEG", quality=90)
        cnt_arr += [new_file_path]
    print("Rotating file : %s - %s " % (x, d))
    return cnt_arr


readf()
