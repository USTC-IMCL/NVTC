import os, io, glob, time, pickle
import lmdb, random, shutil
import numpy as np
from PIL import Image


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def checkImageIsValid(input_size, minsize):
    w, h = input_size
    if w * h == 0 or w < minsize[0] or h < minsize[1]:
        return False
    return True

def get_crop_params(input_size, output_size):
    w, h = input_size
    tw, th = output_size
    if w < tw or h < th:
        print(w, h)
        return 0, min(w, tw), 0, min(h, th)
    w1 = random.randint(0, w - tw)
    h1 = random.randint(0, h - th)
    w2 = w1 + tw
    h2 = h1 + th
    return w1, w2, h1, h2

def img_to_string(img):
    f = io.BytesIO()
    img.save(f, 'PNG', compress_level=1)
    return f.getvalue()

def string_to_img(string):
    img = Image.open(io.BytesIO(string))
    img = img.convert('RGB')
    return img


def create_dataset_img(lmdb_dir, img_ps, epoch=1, crop_size=None, minsize=(256, 256)):
    shutil.rmtree(lmdb_dir, ignore_errors=True)
    os.makedirs(lmdb_dir, exist_ok=True)

    nSamples = len(img_ps)
    env = lmdb.open(lmdb_dir, map_size=1099511627776)
    cache = {}
    cnt = 1
    for ie in range(epoch):
        for i in range(nSamples):
            img_p = img_ps[i]
            if not os.path.exists(img_p):
                nSamples -= 1
                print('%s does not exist' % img_p)
                continue
            with open(img_p, 'rb') as f:
                img_bin = f.read()
            if img_bin is None:
                nSamples -= 1
                continue

            img = string_to_img(img_bin)
            if not checkImageIsValid(img.size, minsize):
                nSamples -= 1
                continue

            if crop_size:
                w1, w2, h1, h2 = get_crop_params(img.size, crop_size)
                img = img.crop((w1, h1, w2, h2))
                img_bin = img_to_string(img)

            cropiKey = 'img-{:0>9}'.format(cnt)
            cache[cropiKey.encode()] = img_bin

            if cnt % 100 == 0:
                writeCache(env, cache)
                cache = {}
                print('Written %d / %d' % (cnt, nSamples))
            cnt += 1

    nSamples = cnt - 1
    cache[b'num-samples'] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with {} samples'.format(nSamples))


if __name__ == '__main__':
    img_ps = sorted(glob.glob('/data/datasets/coco/train2017/*.jpg'))
    lmdb_dir = '/data/datasets/coco_train2017_lmdb'
    create_dataset_img(lmdb_dir, img_ps, epoch=1, minsize=(512, 512))



