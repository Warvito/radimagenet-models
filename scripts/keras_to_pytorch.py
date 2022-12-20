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


def convert_conv(pytorch_conv, tf_conv):
    pytorch_conv.weight.data = torch.tensor(np.transpose(tf_conv.kernel.numpy(), (3, 2, 0, 1)))
    pytorch_conv.bias.data = torch.tensor(tf_conv.bias.numpy())
    return pytorch_conv


def convert_bn(pytorch_bn, tf_bn):
    pytorch_bn.weight.data = torch.tensor(tf_bn.gamma.numpy())
    pytorch_bn.bias.data = torch.tensor(tf_bn.beta.numpy())
    pytorch_bn.running_mean.data = torch.tensor(tf_bn.moving_mean.numpy())
    pytorch_bn.running_var.data = torch.tensor(tf_bn.moving_variance.numpy())
    return pytorch_bn


pytorch_model = ResNet50()
tf_keras_model = tf.keras.models.load_model(input_path)

pytorch_model.conv1 = convert_conv(pytorch_model.conv1, tf_keras_model.get_layer("conv1_conv"))
pytorch_model.bn1 = convert_bn(pytorch_model.bn1, tf_keras_model.get_layer("conv1_bn"))



conv_name = "conv2_"
layers_list = []
for layer in tf_keras_model.layers:
    if conv_name in layer.get_config()['name']:
        layers_list.append(layer)

for i in range(1, 4):
    for j in range(0, 4):
        for layer in layers_list:
            if f"{conv_name}block{str(i)}_{str(j)}" in layer.get_config()['name']:
                print(layer.get_config()['name'])


pytorch_model.layer1[0].get_submodule("bn1")

pytorch_model.conv1.weight.data = torch.tensor(np.transpose(tf_keras_model.get_layer("conv1_conv").kernel.numpy(), (3, 2, 0, 1)))
pytorch_model.conv1.bias.data = torch.tensor(tf_keras_model.get_layer("conv1_conv").bias.numpy())

pytorch_model.bn1.weight.data = torch.tensor(tf_keras_model.get_layer("conv1_bn").gamma.numpy())
pytorch_model.bn1.bias.data = torch.tensor(tf_keras_model.get_layer("conv1_bn").beta.numpy())
pytorch_model.bn1.running_mean.data = torch.tensor(tf_keras_model.get_layer("conv1_bn").moving_mean.numpy())
pytorch_model.bn1.running_var.data = torch.tensor(tf_keras_model.get_layer("conv1_bn").moving_variance.numpy())