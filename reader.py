import numpy as np
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import torch
import json
import random
import copy


# randomly move the bounding box around
def aug_line(line, width, height):
    bbox = np.array(line[2:5])
    bias = round(30 * random.uniform(-1, 1))
    bias = max(np.max(-bbox[0, [0, 2]]), bias)
    bias = max(np.max(-2*bbox[1:, [0, 2]]+0.5), bias)
    
    line[2][0] += int(round(bias))
    line[2][1] += int(round(bias))
    line[2][2] += int(round(bias))
    line[2][3] += int(round(bias))

    line[3][0] += int(round(0.5*bias))
    line[3][1] += int(round(0.5*bias))
    line[3][2] += int(round(0.5*bias))
    line[3][3] += int(round(0.5*bias))

    line[4][0] += int(round(0.5*bias))
    line[4][1] += int(round(0.5*bias))
    line[4][2] += int(round(0.5*bias))
    line[4][3] += int(round(0.5*bias))

    line[5][2] = line[2][2]/width
    line[5][3] = line[2][0]/height

    line[5][6] = line[3][2]/width
    line[5][7] = line[3][0]/height

    line[5][10] = line[4][2]/width
    line[5][11] = line[4][0]/height
    return line

class loader(Dataset):

    def __init__(self, data_path, data_type):
        self.lines = []
        self.labels = {}
        self.data_path = data_path
        self.data_type = data_type
        subjects = os.listdir(data_path)
        # 排序
        subjects.sort()
        # 遍历每一个参与者文件夹
        for subject in subjects:
            subject_path = os.path.join(data_path, subject)
            if ((not os.path.isdir(subject_path)) or subject=='01185' or subject=='01730' or subject=='02065'):
                continue
            # info.json 文件内容如下：
            #   TotalFrames: 这个对象的总帧数。
            #   NumFaceDetections: 检测到人脸的帧数。
            #   NumEyeDetections: 检测到眼睛的帧数。
            #   Dataset. "train "，" val "，或 "test "。
            #   DeviceName: 在记录中使用的设备的名称。如 iPhone 6 等
            info_json = json.load(open(os.path.join(subject_path, "info.json"), "r"))
            # 获取数据类型，可能的取值为 train、test、val
            current_data_type = info_json["Dataset"]
            # 获取设备名，
            device_name = info_json["DeviceName"]
            # dotInfo.json 文件内容如下：
            #   DotNum: 在该帧中显示的点序列号(从 0 开始)。
            #   XPts, YPts: 点的中心距屏幕左上角的位置。
            #   XCam, YCam: 点的中心在预测空间（摄像机在 (0,0) 处）中的位置。
            #   Time: 显示的点第一次出现在屏幕上的时间(秒)。
            self.labels[subject] = json.load(open(os.path.join(subject_path, "dotInfo.json"), "r"))
            if not current_data_type == data_type:
                continue

            # test on iPhone datas only
            # if data_type=="test" and device_name[:6]!="iPhone":
            #     continue
            # newFaceLdmk.json 为该项目新增的标签文件。文件内容如下：
            #
            name_file = open(os.path.join(subject_path, "newFaceLdmk.json"), "r")
            temp = json.load(name_file)

            self.lines = self.lines + temp
            # if (len(self.lines)>=150000):
            #    break


    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        #   0        1     2     3     4     5     6
        # subject, name, face, left, right, rect, 8pts
        
        line = self.lines[idx]

        # 拼接参与者路径，如 D:\datasets\GazeCapture\00002
        subject_path = os.path.join(self.data_path, line[0])
        # 获取帧名，如 00000.jpg
        frame = line[1]
        # 帧名去掉 .jpg 后缀，并转为数字类型
        item_index = int(frame[:-4])
        # 读取帧图像
        img = cv2.imread(os.path.join(subject_path, "frames", frame))
        img = np.array(img)
        # 获取帧的宽、高
        height = img.shape[0]
        width = img.shape[1]

        origin = copy.deepcopy(line)
        # print(line[2])
        if not (self.data_type == 'test'):
           line = aug_line(copy.deepcopy(line), width, height)

        # 截取脸部、左眼、右眼图像
        face_img = img[line[2][0]:line[2][1], line[2][2]:line[2][3]]
        leftEye_img = img[line[3][0]:line[3][1], line[3][2]:line[3][3]]
        rightEye_img = img[line[4][0]:line[4][1], line[4][2]:line[4][3]]

        # 将脸部图像的尺寸调整为 224*224
        face_img = cv2.resize(face_img, (224, 224))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = face_img / 255
        face_img = face_img.transpose(2, 0, 1)


        # 将眼睛图像的尺寸调整为 112*112
        leftEye_img = cv2.resize(leftEye_img, (112, 112))
        leftEye_img = cv2.cvtColor(leftEye_img, cv2.COLOR_BGR2RGB)
        leftEye_img = leftEye_img / 255
        leftEye_img = leftEye_img.transpose(2, 0, 1)

        rightEye_img = cv2.resize(rightEye_img, (112, 112))
        rightEye_img = cv2.cvtColor(rightEye_img, cv2.COLOR_BGR2RGB)
        rightEye_img = cv2.flip(rightEye_img,1)
        rightEye_img = rightEye_img / 255
        rightEye_img = rightEye_img.transpose(2, 0, 1)


       
        rects = line[5]
        ex_label = line[6]
        label = np.array([self.labels[line[0]]["XCam"][item_index],
                          self.labels[line[0]]["YCam"][item_index]])

        return {"faceImg": torch.from_numpy(face_img).type(torch.FloatTensor), 
                "leftEyeImg": torch.from_numpy(leftEye_img).type(torch.FloatTensor),
                "rightEyeImg": torch.from_numpy(rightEye_img).type(torch.FloatTensor),
                "rects": torch.from_numpy(np.array(rects)).type(torch.FloatTensor),
                "label": torch.from_numpy(label).type(torch.FloatTensor), 
                "exlabel": torch.from_numpy(np.array(ex_label)).type(torch.FloatTensor),
                "frame": line}


def txtload(path, type, batch_size, shuffle=False, num_workers=0):
    dataset = loader(path, type)
    print("[Read Data]: GazeCapture Dataset")
    print("[Read Data]: Total num: {:d}".format(len(dataset)))
    print("[Read Data]: Dataset type: {:s}".format(type))
    load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return load


if __name__ == "__main__":
    path = "/home/byw/Dataset/GazeCapture/"
    type = "train"
    loader = txtload(path, type, batch_size=2)
    for i, (data) in enumerate(loader):
        #print(data['frame'][0][0] + ' ' + data['frame'][1][0])
        '''print(data['faceImg'][0].shape)
                                print(torch.mean(data['faceImg'][0]))
                                print(torch.mean(data['leftEyeImg'][0]))
                                print(torch.mean(data['rightEyeImg'][0]))
                                print(data['rects'][0])
                                print(data['exlabel'][0])'''
        break
