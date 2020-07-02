import cv2
import os
import numpy as np

def resize (brand, filename):
    # for filename in os.listdir('./data_resized/test/aesop/'):
        img = cv2.imread('./' + brand + '/' + filename, cv2.IMREAD_UNCHANGED)
        if img is None:
            print('Exception occurred: ', filename)
            return []

        print('Original Dimensions: ', img.shape)

        # scale_percent = 60
        # width = int(img.shape[1] * scale_percent / 100)
        # height = int(img.shape[0] * scale_percent / 100)
        width = 300
        height = 300
        dim = (width, height)

        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return resized

        # cv2.imwrite('./data_resized/test/aesop/' + filename, resized)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

for filename in os.listdir('./aesop/'):
    resized = resize('aesop', filename)
    if len(resized) == 0:
        continue
    if np.random.rand(1) < 0.2:
        cv2.imwrite('./data_resized/validation/aesop/' + filename, resized)
    else:
        cv2.imwrite('./data_resized/train/aesop/' + filename, resized)

for filename in os.listdir('./kiehls/'):
    resized = resize('kiehls', filename)
    if len(resized) == 0:
        continue
    if np.random.rand(1) < 0.2:
        cv2.imwrite('./data_resized/validation/kiehls/' + filename, resized)
    else:
        cv2.imwrite('./data_resized/train/kiehls/' + filename, resized)