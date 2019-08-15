import matplotlib.pyplot as plt
import numpy as np
from keras import applications
from keras.models import Model
from scipy.misc import imread
from scipy.misc import imresize

import model as GP

# dimensions of our images.
img_width, img_height = 448, 448

class_num = 200


def preprocess_image_batch(image_paths, image_num=1, img_size=None, crop_size=None, color_mode='rgb', out=None):
    """
    Consistent preprocessing of images batches

    :param image_paths: iterable: images to process
    :param crop_size: tuple: crop images if specified
    :param img_size: tuple: resize images if specified
    :param color_mode: Use rgb or change to bgr mode based on type of model you want to use
    :param out: append output to this iterable if specified
    """
    img_list = []

    for im_path in image_paths:
        img = imread(im_path, mode='RGB')
        if img_size:
            img = imresize(img, img_size)

        img = img.astype('float32')
        if color_mode == 'bgr':
            img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]

        if crop_size:
            img = img[:, (img_size[0] - crop_size[0]) // 2:(img_size[0] + crop_size[0]) // 2
            , (img_size[1] - crop_size[1]) // 2:(img_size[1] + crop_size[1]) // 2]

        img_list.append(img)

    try:
        img_batch = np.stack(img_list, axis=0)
    except:
        raise ValueError('when img_size and crop_size are None, images'
                         ' in image_paths must have the same shapes.')

    if out is not None and hasattr(out, 'append'):
        out.append(img_batch)
    else:
        return [img_batch] * image_num


def gatp_two_stream(img):
    # build the network
    modelvgg = applications.InceptionV3(weights=None, include_top=False, input_shape=(img_width, img_height, 3))
    modelvgg2 = applications.InceptionV3(weights=None, include_top=False, input_shape=(img_width, img_height, 3))

    print('Model loaded.')

    model = GP.build_global_attention_pooling_model_cascade_attention([modelvgg, modelvgg2], class_num)

    model.load_weights('weights1/weights-gatp-two-stream-inception_v3-006-0.9080.hdf5')

    model = Model(inputs=model.inputs, outputs=model.get_layer('mask').output)

    return model.predict(img)


if __name__ == '__main__':
    img_path = 'CUB_200_2011/images/075.Green_Jay/Green_Jay_0040_65863.jpg'
    img = preprocess_image_batch([img_path], image_num=2, img_size=(img_height, img_width))
    heatmap = gatp_two_stream(img)
    heatmap = heatmap[0, :, :, 0]

    image = plt.imread(img_path)
    heatmap = imresize(heatmap, np.shape(image)[:2], interp='nearest')

    plt.imshow(image)
    plt.imshow(heatmap, alpha=0.5, cmap="RdBu")
    plt.show()
