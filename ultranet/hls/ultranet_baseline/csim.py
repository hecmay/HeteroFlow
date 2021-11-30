import os
import sys
import argparse
import math

import cv2
import numpy as np
import torch

sys.path.append('../..')
from models.model import create_model

def square_sum_error(list_a, list_b):
    if len(list_a) != len(list_b):
        raise ValueError("List size not equal!\n")
    sse = 0.0
    for i in range(len(list_a)):
        sse += (list_a[i] - list_b[i]) ** 2
    return sse

def mean(list):
    sum = 0.0
    for i in range(len(list)):
        sum += list[i]
    return sum / len(list)

def stddev(list):
    m = mean(list)
    return math.sqrt(square_sum_error(list, [m]*len(list)))

def relative_error(list_a, list_b):
    if len(list_a) != len(list_b):
        raise (ValueError, "List size not equal!\n")
    re = []
    for i in range(len(list_a)):
        if list_b[i] != 0 and list_a[i] != 0:
            re.append(list_a[i] / list_b[i])
    return mean(re), stddev(re)

def diff_log(actual, reference, filename):
    if len(actual) != len(reference):
        raise (ValueError, "List size not equal!\n")
    with open(filename, 'w') as f:
        for i in range(len(actual)):
            if actual[i] != reference[i]:
                f.write("Mismatch at i = " + str(i) + ", Expected = " + str(reference[i]) + ", Actual = " + str(actual[i]) + "\n")

def save_results(layername):
    if layername == 'after_resize':
        drange = 255.0
        last = False
    elif layername == 'result':
        drange = 105.0
        last = True
    else:
        drange = 2 ** HLS_oshapes[layername][3] - 1
        last = False
    result_shape = HLS_oshapes[layername]
    result_tensor = pth_outputs[HLS_layers[layername][0]]
    result = result_tensor.detach().numpy()[0]
    last_bias = model.yolo_conv.bias.detach().numpy()
    with open(result_dir + "/pytorch_" + layername + ".dat", 'w') as f:
        for row in range(result_shape[0]):
            for col in range(result_shape[1]):
                for ch in range(result_shape[2]):
                    if last:
                        val = int((result[ch][row][col] - last_bias[ch]) * drange)
                    else:
                        val = int(result[ch][row][col] * drange)
                    f.write(str(val))
                    f.write('\n')

def compare_results(layername):
    hls_result_file = result_dir + "/xcel_" + layername + ".dat"
    pytorch_result_file = result_dir + "/pytorch_" + layername + ".dat"
    hls_r = []
    pth_r = []
    with open(hls_result_file, 'r') as hls_f:
        for line in hls_f:
            hls_r.append(int(line[:-1])) # remove the "\n"
    with open(pytorch_result_file, 'r') as pth_f:
        for line in pth_f:
            pth_r.append(int(line[:-1])) # remove the "\n"
    sse = square_sum_error(hls_r, pth_r)
    mse = sse / len(hls_r)
    # mre, dre = relative_error(hls_r, pth_r)
    print("Layer:", layername, "output size", len(hls_r), end=' | ')
    # print("SSE", sse, "MSE", mse, "MRE", mre, "DRE", dre)
    print("SSE", sse, "MSE", mse)
    diff_log(hls_r, pth_r, result_dir + "/diff_" + layername + ".dat")

IMG_H = 360
IMG_W = 640
IN_CH = 3
RESIZE_H = 320
RESIZE_W = 640

parser = argparse.ArgumentParser()
parser.add_argument('img_path', type=str)
parser.add_argument('result_dir', type=str)
img_path = parser.parse_args().img_path
result_dir = parser.parse_args().result_dir
hls_img_path = result_dir + "/xcel_before_resize.dat"

model = create_model('yolo', 'ultra', 'dorefa', 4, 4)
state_dict = torch.load('../../export/4bitfsnp2.pt', map_location = torch.device('cpu'))['model']

model.load_state_dict(state_dict)
model.eval()
# print(model)

# add hook to every layer whose output exists in HLS
pth_outputs= []
def hook(module, input, output):
    pth_outputs.append(output)

# HLS_layers[key][0] is the index in the output[] list
# HLS_layers[key][1] is the layer object
has_cq = 0
HLS_layers = {
    "conv0"  :   (0, model.backbone.layers[has_cq + 2]),
    "pool0"  :   (1, model.backbone.layers[has_cq + 3]),
    "conv1"  :   (2, model.backbone.layers[has_cq + 6]),
    "pool1"  :   (3, model.backbone.layers[has_cq + 7]),
    "conv2"  :   (4, model.backbone.layers[has_cq + 10]),
    "pool2"  :   (5, model.backbone.layers[has_cq + 11]),
    "conv3"  :   (6, model.backbone.layers[has_cq + 14]),
    "pool3"  :   (7, model.backbone.layers[has_cq + 15]),
    "conv4"  :   (8, model.backbone.layers[has_cq + 18]),
    "conv5"  :   (9, model.backbone.layers[has_cq + 21]),
    "conv6"  :   (10, model.backbone.layers[has_cq + 24]),
    "conv7"  :   (11, model.backbone.layers[has_cq + 27]),
    "result" :   (12, model.yolo_conv),
}
for layername in HLS_layers.keys():
    HLS_layers[layername][1].register_forward_hook(hook)

# all HLS layers' output shapes, noted in (H,W,C,B)
HLS_oshapes = {
    "conv0"  :   (320, 640, 16, 4),
    "pool0"  :   (160, 320, 16, 4),
    "conv1"  :   (160, 320, 32, 4),
    "pool1"  :   (80,  160, 32, 4),
    "conv2"  :   (80,  160, 64, 4),
    "pool2"  :   (40,  80,  64, 4),
    "conv3"  :   (40,  80,  64, 4),
    "pool3"  :   (20,  40,  64, 4),
    "conv4"  :   (20,  40,  64, 4),
    "conv5"  :   (20,  40,  64, 4),
    "conv6"  :   (20,  40,  64, 4),
    "conv7"  :   (20,  40,  64, 4),
    "result" :   (20,  40,  36, 32),
}


image = cv2.imread(img_path)
print('loaded image with shape:', image.shape)

# dump input image to hls
with open(hls_img_path,'w') as f:
    for row in range(IMG_H) :
        for col in range(IMG_W):
            for ch in range(IN_CH):
                f.write(str(image[row][col][ch]))
                f.write('\n')

# save image for compare
with open(result_dir + '/pytorch_before_resize.dat', 'w') as f:
    for row in range(IMG_H):
        for col in range(IMG_W):
            for ch in range(IN_CH):
                f.write(str(image[row][col][ch]))
                f.write('\n')

# resize input and save the resized image
image = cv2.resize(image, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_LINEAR)
image_np = np.array(image)
with open(result_dir + '/pytorch_after_resize.dat', 'w') as f:
    for row in range(RESIZE_H):
        for col in range(RESIZE_W):
            for ch in range(IN_CH):
                f.write(str(image_np[row][col][ch]))
                f.write('\n')

# invoke the c model
cmd = './ultranet_csim/ultranet_csim_exe ' + hls_img_path + ' ' + result_dir
print(cmd)
os.system(cmd)

# invoke the pytorch model
image_np = image_np.transpose(2, 0, 1)
image_tensor = torch.from_numpy(image_np)
image_tensor = image_tensor.float() / 255.0
model.forward(image_tensor[None,...])

# save python results
for layername in HLS_layers.keys():
    save_results(layername)

# find the dinamic range of the yolo conv output
# yolo_conv_out_tensor = pth_outputs[HLS_layers['result'][0]]
# yolo_conv_out_drange = torch.max(torch.abs(yolo_conv_out_tensor))
# print(yolo_conv_out_drange)

# read hls results and compare
compare_results("before_resize")
compare_results("after_resize")
for layername in HLS_layers.keys():
    compare_results(layername)

