#align_by_5p
import numpy as np
import cv2
import glob
import os
import argparse

"""根据五点对齐人脸
输入：
image_wild_dir：储存没有对齐的人脸的文件夹
lm_dir：储存人脸关键点文件的文件夹
target_dir： 用来存储对齐后人脸的文件夹
"""

parser = argparse.ArgumentParser(description='Argument parser')

parser.add_argument('--image_wild_dir', dest='image_wild_dir',
                    default="E:\\Zhiyang\\test\\sunglass",
                    help='path to image unaligned')

parser.add_argument('--lm_dir', dest='lm_dir',
                    default="E:\\Zhiyang\\test\\sunglasslandmark",
                    help='path to landmark files')

parser.add_argument('--target_dir', dest='target_dir',
                    default="E:\\Zhiyang\\test\\result",
                    help='path to save aligned images')

parser.add_argument('--IS_VISUAL', dest='IS_VISUAL', type=bool,
                    default=0,
                    help='IS_VISUAL?')

args = parser.parse_args()

def mkdir_if_missing(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 平均人脸关键点位置
mean_face_lm5p = np.array([
    [-0.17607, -0.172844],  # 左眼球
    [0.1736, -0.17356],  # 右眼球
    [-0.00182, 0.0357164],  # 鼻尖
    [-0.14617, 0.20185],  # 左嘴角
    [0.14496, 0.19943],  # 右嘴角
])


def _get_align_5p_mat23_size_256(lm):

    width = 256
    mf = mean_face_lm5p.copy()

    # 1. 输出图片大小是256*256
    # 2. 两眼球之间距离约为70像素
    ratio = 70.0 / (
        256.0 * 0.34967
    )  # magic number 0.34967 to compensate scaling from average landmarks

    left_eye_pupil_y = mf[0][1]
    # 计算仿射变换矩阵，对齐图片中，人眼距离上下边界距离比例为 1:1.42
    ratioy = (left_eye_pupil_y * ratio + 0.5) * (1 + 1.42)
    mf[:, 0] = (mf[:, 0] * ratio + 0.5) * width
    mf[:, 1] = (mf[:, 1] * ratio + 0.5) * width / ratioy
    mx = mf[:, 0].mean()
    my = mf[:, 1].mean()
    dmx = lm[:, 0].mean()
    dmy = lm[:, 1].mean()
    mat = np.zeros((3, 3), dtype=float)
    ux = mf[:, 0] - mx
    uy = mf[:, 1] - my
    dux = lm[:, 0] - dmx
    duy = lm[:, 1] - dmy
    c1 = (ux * dux + uy * duy).sum()
    c2 = (ux * duy - uy * dux).sum()
    c3 = (dux ** 2 + duy ** 2).sum()
    a = c1 / c3
    b = c2 / c3

    kx = 1
    ky = 1

    s = c3 / (c1 ** 2 + c2 ** 2)
    ka = c1 * s
    kb = c2 * s

    transform = np.zeros((2, 3))
    transform[0][0] = kx * a
    transform[0][1] = kx * b
    transform[0][2] = mx - kx * a * dmx - kx * b * dmy
    transform[1][0] = -ky * b
    transform[1][1] = ky * a
    transform[1][2] = my - ky * a * dmy + ky * b * dmx
    return transform


def get_align_5p_mat23(lm5p, size):
    """用5个关键点对齐人脸
    左眼球, 右眼球, 鼻尖, 左嘴角, 右嘴角

    lm5p: nparray of (5, 2), 5 个人脸关键点,

    size: an integer, 输出图像大小

    返回: 放射变换矩阵 of shape (2, 3)
    """
    mat23 = _get_align_5p_mat23_size_256(lm5p.copy())
    mat23 *= size / 256
    return mat23


def align_given_lm5p(img, lm5p, size):
    mat23 = get_align_5p_mat23(lm5p, size)
    return cv2.warpAffine(img, mat23, (size, size))


def align_face_5p(img, landmarks):
    aligned_img = align_given_lm5p(img, np.array(landmarks).reshape((5, 2)), 256)
    return aligned_img


def main():
    IS_VISUAL = args.IS_VISUAL  # 是否可视化调试

    image_wild_dir = args.image_wild_dir  # "E:\\Zhiyang\\project\\datasets\\yancongquan\\sunglass"
    lm_dir = args.lm_dir  # 'E:\\Zhiyang\\project\\datasets\\yancongquan\\sunglasslandmark'
    target_dir = args.target_dir  # "E:\\Zhiyang\\project\\datasets\\yancongquan\\aligned-sunglass"

    lm_point = [(92, 106), (163, 106), (127, 153), (98, 191), (157, 190)]  # 大概对齐后关键点的位置
    mkdir_if_missing(target_dir)
    file_list = glob.glob(os.path.join(image_wild_dir, '*.jpg'))
    lm_list = list(map(lambda x: x.replace(image_wild_dir, lm_dir), file_list))
    lm_list = list(map(lambda x: x.replace('.jpg', '.txt'), lm_list))

    target_list = list(map(lambda x: x.replace(image_wild_dir, target_dir), file_list))

    for i in range(len(file_list)):
        if i % 1000 == 0:
            print("processing {} / {}".format(i, len(file_list)))
        img = cv2.imread(file_list[i], cv2.IMREAD_UNCHANGED)
        with open(lm_list[i], 'r') as f:
        # with open('./demo/landmark1.txt', 'r') as f:
            landmarks = list(map(float, f.read().strip().split(' ')))
        try:
            aligned_img_5p = align_face_5p(img=img, landmarks=landmarks)
            if IS_VISUAL:
                for idx in range(5):
                    cv2.circle(aligned_img_5p, lm_point[idx], 5, (0, 0, 255), -1)
                cv2.imshow('1', img)
                cv2.imshow('2', aligned_img_5p)
                cv2.waitKey(0)

            cv2.imwrite(target_list[i], aligned_img_5p)

        except Exception as err:
            print("fail to save image "+file_list[i])

if __name__ == '__main__':
    main()
