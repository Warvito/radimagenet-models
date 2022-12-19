"""
Script based on https://github.com/BMEII-AI/RadImageNet/issues/3#issuecomment-1232417600
and https://discuss.pytorch.org/t/transferring-weights-from-keras-to-pytorch/9889
"""
import tensorflow as tf
import numpy as np
import torch
from tensorflow import keras
from radimagenet_models.models.resnet import ResNet50

input_path = '/media/walter/Storage/Projects/radimagenet-models/outputs/original_weights/RadImageNet-ResNet50_notop.h5'
out_path = '/media/walter/Storage/Projects/radimagenet-models/outputs/pytorch_weights/RadImageNet-ResNet50_notop.pt'

def keras_to_pytorch(keras_model):
    weight_dict = {
        "conv_weights": dict(),
        "conv_bias": dict(),
        "bn_gamma": dict(),
        "bn_beta": dict(),
        "bn_moving_mean": dict(),
        "bn_moving_variance": dict(),
    }
    for layer in keras_model.layers:
        print(layer.get_config()['name'])
        if type(layer) is keras.layers.Conv2D:
            weight_dict["conv_weights"][layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0], (3, 2, 0, 1))
            weight_dict["conv_bias"][layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
        elif type(layer) is keras.layers.BatchNormalization:
            weight_dict["bn_gamma"][layer.get_config()['name'] + '.gamma'] = layer.gamma
            weight_dict["bn_beta"][layer.get_config()['name'] + '.beta'] = layer.beta
            weight_dict["bn_moving_mean"][layer.get_config()['name'] + '.moving_mean'] = layer.moving_mean
            weight_dict["bn_moving_variance"][layer.get_config()['name'] + '.moving_variance'] = layer.moving_variance
    return weight_dict


pytorch_model = ResNet50()
tf_keras_model = tf.keras.models.load_model(input_path)
weights = keras_to_pytorch(tf_keras_model)



i = 0
j = 0
for name, param in pytorch_model.named_parameters():
    if 'conv' in name:
        # param.data = torch.tensor(values[i])
        i += 1
    if 'bn' in name:
        if 'weight' in name:
        print("ou")

for layer in pytorch_model.children():
    print(layer)

x = torch.ones((2,3,224,224))
outputs = pytorch_model(x)