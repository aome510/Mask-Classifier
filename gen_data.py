import os
import cv2
import random
import numpy as np
import scipy.io


def gen_data_mask_train(min_size=500):
    try:
        print('gen train masks')

        os.system('mkdir ./data/MAFA/train_masks/')
        os.system(
            'find ./data/MAFA/train_masks/ -name "*" -type f -exec rm {} \\;')

        mat = scipy.io.loadmat(
            './data/MAFA' + '/MAFA-Label-Train/LabelTrainAll.mat')['label_train']
        n_image_train = mat.shape[1]

        id = 0
        f = open('./data/MAFA' + '/train_masks/imglist.txt', 'w')
        for i in range(n_image_train):

            img_name = mat[0][i][1][0]
            img_data = mat[0][i][2]

            img_arr = cv2.imread('./data/MAFA' +
                                 '/train-images/images/' + img_name)

            for j in img_data:
                j = j.astype(int)
                [x, y, w, h] = j[0:4]
                occ_type = j[12]
                occ_degree = j[13]

                if w * h <= min_size or w <= 0 or h <= 0 \
                        or y + h >= img_arr.shape[0] \
                        or x + w >= img_arr.shape[1]:
                    continue

                id += 1
                if id % 1000 == 0:
                    print(id, '...')
                img_path = './data/MAFA' + '/train_masks/train_mask_' + \
                    str(id).zfill(5) + '.jpg'
                cv2.imwrite(img_path, img_arr[y:y+h, x:x+w])

                if occ_type != 3 and occ_degree >= 2:
                    f.write(img_path + ' 0\n')
                else:
                    f.write(img_path + ' 1\n')

        f.close()

    except Exception as e:
        # print('Error:', e)
        print('Please download MAFA Dataset')


def gen_data_mask_test(min_size=500):
    try:
        print('gen test masks')

        os.system('mkdir ./data/MAFA/test_masks/')
        os.system(
            'find ./data/MAFA/test_masks/ -name "*" -type f -exec rm {} \\;')

        mat = scipy.io.loadmat(
            './data/MAFA' + '/MAFA-Label-Test/LabelTestAll.mat')['LabelTest']
        n_image_test = mat.shape[1]

        id = 0
        f = open('./data/MAFA' + '/test_masks/imglist.txt', 'w')
        for i in range(n_image_test):

            img_name = mat[0][i][0][0]
            img_data = mat[0][i][1]

            img_arr = cv2.imread('./data/MAFA' +
                                 '/test-images/images/' + img_name)

            for j in img_data:
                j = j.astype(int)
                [x, y, w, h] = j[0:4]
                face_type = j[4]
                occ_type = j[9]
                occ_degree = j[10]

                if w * h <= min_size or w <= 0 or h <= 0 \
                        or face_type == 3 \
                        or y + h >= img_arr.shape[0] \
                        or x + w >= img_arr.shape[1]:
                    continue

                id += 1
                if id % 1000 == 0:
                    print(id, '...')
                img_path = './data/MAFA' + '/test_masks/test_mask_' + \
                    str(id).zfill(5) + '.jpg'

                cv2.imwrite(img_path, img_arr[y:y+h, x:x+w])
                if face_type == 1 and occ_degree >= 2 and occ_type != 3:
                    f.write(img_path + ' 0\n')
                else:
                    f.write(img_path + ' 1\n')

        f.close()

    except Exception as e:
        # print('Error:', e)
        print('Please download MAFA Dataset')


def gen_data(name):
    print('gen data {}'.format(name))

    files = open('./data/{}.txt'.format(name), 'r').readlines()

    # create neccessary folders
    os.system('mkdir ./data/{}/'.format(name))
    os.system('mkdir ./data/{}/mask/'.format(name))
    os.system('mkdir ./data/{}/nomask/'.format(name))

    # remove all files
    os.system(
        'find ./data/{}/mask/ -name "*" -type f -exec rm {{}} \\;'.format(name))
    os.system(
        'find ./data/{}/nomask/ -name "*" -type f -exec rm {{}} \\;'.format(name))

    # add files
    cnt_mask = 0
    cnt_nomask = 0
    for file in files:
        file = file.strip().split(' ')
        path = file[0]
        label = file[1]

        if (cnt_mask + cnt_nomask) % 1000 == 0:
            print(name, cnt_mask, cnt_nomask, '.....')

        if label == '0':  # mask
            cnt_mask += 1
            os.system('cp {} ./data/{}/mask/{}.jpg'
                      .format(path, name, name + '_mask_' + str(cnt_mask).zfill(5)))
        else:  # nomask
            cnt_nomask += 1
            os.system('cp {} ./data/{}/nomask/{}.jpg'
                      .format(path, name, name + '_nomask_' + str(cnt_nomask).zfill(5)))


def gen_data_nomask(dir, n_img):
    files = [
        (dir + file + ' 1')
        for file in os.listdir(dir)
        if file.endswith('.jpg')
    ]
    random.shuffle(files)
    files = files[:n_img]

    f = open(dir + 'imglist.txt', 'w')
    f.write('\n'.join(files) + '\n')
    f.close()


def label(in_path, out_path, dir, start_id):
    in_labels = open(in_path, 'r').readlines()
    out_labels = open(out_path, 'w')

    for label in in_labels:
        label = label.strip().split(' ')

        if len(label) == 2:
            out_labels.write(dir + label[0] + ' ' + label[1] + '\n')
        else:
            out_labels.write(dir + str(start_id).zfill(5)
                             + '.jpg ' + label[0] + '\n')
            start_id += 1

    out_labels.close()


def label_from_dir(dir):
    files = os.listdir(dir + 'labels/raw/')

    for file in files:
        file_info = file.split('.')[0].split('-')

        if len(file_info) == 1:
            start_id = 1
        else:
            start_id = int(file_info[1])

        label(dir + 'labels/raw/' + file, dir + 'labels/' + file,
              dir + file_info[0] + '/', start_id)


def gen_data_mask_classifier():
    try:
        print('gen data mask classifier')

        dir = './data/mask_classifier/'
        label_from_dir(dir)

        os.system('cat {} > {}'.format(dir + '/labels/*.txt', dir + 'imglist.txt'))

        files = open(dir + 'imglist.txt', 'r').readlines()
        train = open(dir + 'imglist_train.txt', 'w')
        test = open(dir + 'imglist_test.txt', 'w')

        files = [
            file for file in files
            if not file.endswith('2\n')
        ]

        random.shuffle(files)
        n_files = len(files)
        n_train = int(0.8 * n_files)
        train.write(''.join(files[:n_train]))
        test.write(''.join(files[n_train:n_files]))

        train.close()
        test.close()
    except Exception as e:
        # print('Error:', e)
        print('Please download Mask Classifier dataset')


def gen_data_widerface(n_img, min_size=500):
    try:
        print('gen data widerface')

        out_dir = './data/WiderFace_modified/'
        os.system('mkdir {}'.format(out_dir))
        os.system('find {} -name "*" -type f -exec rm {{}} \\;'.format(out_dir))

        dir = './data/WiderFace/'
        imgs = os.listdir(dir)
        random.shuffle(imgs)

        id = 0
        for img_name in imgs:
            img = cv2.imread(dir + img_name)

            if id == n_img:
                break
            if img is None or img.shape[0] * img.shape[1] < min_size:
                continue

            id += 1
            if id % 1000 == 0:
                print(id, '...')
            cv2.imwrite(out_dir + str(id).zfill(5) + '.jpg', img)

        gen_data_nomask(out_dir, n_img)

    except Exception as e:
        # print('Error:', e)
        print('Please download WiderFace dataset')


def gen_data_celebA(n_img):
    from func import SSH_init
    from lib.SSH.SSH.test import detect

    try:
        print('gen data celebA')

        out_dir = './data/celebA/faces/'
        os.system('mkdir {}'.format(out_dir))
        os.system('find {} -name "*" -type f -exec rm {{}} \\;'.format(out_dir))

        dir = './data/celebA/img_align_celeba/'
        imgs = os.listdir(dir)
        random.shuffle(imgs)

        net = SSH_init()

        id = 0
        for img_name in imgs:
            img_path = dir + img_name

            img = cv2.imread(img_path)
            bboxs = detect(net, img_path)[0]

            if id == n_img:
                break
            if bboxs.shape[0] == 0:
                continue
            bbox = bboxs.astype(int)[0]

            id += 1
            if id % 1000 == 0:
                print(id, '...')
            cv2.imwrite('{}{}'.format(out_dir, str(id).zfill(5) +
                                      '.jpg'), img[bbox[1]:bbox[3], bbox[0]:bbox[2]])

        gen_data_nomask(out_dir, n_img)

    except Exception as e:
        # print('Error:', e)
        print('Please download celebA dataset')


if __name__ == '__main__':
    gen_data_mask_train()
    gen_data_mask_test()

    gen_data_celebA(16000)
    gen_data_widerface(9000)

    gen_data_mask_classifier()

    os.system('./scripts/gen_data.sh')

    gen_data('train')
    gen_data('test')
