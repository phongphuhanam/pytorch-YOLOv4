import sys
import onnx
import os
import argparse
import numpy as np
import cv2
import onnxruntime
from packaging import version

from tool.utils import *
from tool.darknet2onnx import *


def main(cfg_file, namesfile, weight_file, image_path, batch_size):

    if batch_size <= 0:
        onnx_path_demo = transform_to_onnx(cfg_file, weight_file, batch_size)
    else:
        # Transform to onnx as specified batch size
        transform_to_onnx(cfg_file, weight_file, batch_size)
        # Transform to onnx as demo
        onnx_path_demo = transform_to_onnx(cfg_file, weight_file, 1)

    if version.parse(onnxruntime.__version__) >= version.parse("1.9"):
        providers = onnxruntime.get_available_providers()
        session = onnxruntime.InferenceSession(onnx_path_demo, providers=providers)
    else:
        session = onnxruntime.InferenceSession(onnx_path_demo)
    # session = onnx.load(onnx_path)
    print("The model expects input shape: ", session.get_inputs()[0].shape)

    image_src = cv2.imread(image_path)
    output_img_path = f'{os.path.splitext(onnx_path_demo)[0]}.jpg'
    detect(session, image_src, namesfile,output_img_path)


def detect(session, image_src, namesfile, savename='predictions_onnx.jpg'):
    IN_IMAGE_H = session.get_inputs()[0].shape[2]
    IN_IMAGE_W = session.get_inputs()[0].shape[3]

    # Input
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    print("Shape of the network input: ", img_in.shape)

    # Compute
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: img_in})

    boxes = post_processing(img_in, 0.4, 0.6, outputs)

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(image_src, boxes[0], savename=savename, class_names=class_names)



if __name__ == '__main__':
    # print("Converting to onnx and running demo ...")
    parser = argparse.ArgumentParser(description='Converting to onnx and running demo ...')
    parser.add_argument('cfg_file', help='Path to darknet cfg file')
    parser.add_argument('namesfile', help='Path to darknet name file')
    parser.add_argument('weight_file', help='Path to darknet weight file')
    parser.add_argument('--image_path', help='Path to test image file', default='')
    parser.add_argument('--batch_size', help='onnx batch size', default=-1)
    args = parser.parse_args()
    if not os.path.isfile(args.image_path):
        args.image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/dog.jpg')
    
    main(args.cfg_file, args.namesfile, args.weight_file, args.image_path, args.batch_size)
    
