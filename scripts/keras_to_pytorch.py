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
pytorch_model.eval()
tf_keras_model = tf.keras.models.load_model(input_path)

pytorch_model.conv1 = convert_conv(pytorch_model.conv1, tf_keras_model.get_layer("conv1_conv"))
pytorch_model.bn1 = convert_bn(pytorch_model.bn1, tf_keras_model.get_layer("conv1_bn"))

conv_name = "conv2_"
num_blocks = 3
layers_list = []
for layer in tf_keras_model.layers:
    if conv_name in layer.get_config()['name']:
        layers_list.append(layer)

pytorch_stack = pytorch_model.layer1
for i in range(1, num_blocks+1):
    pytorch_block = pytorch_stack[i-1]
    for layer in layers_list:
        print(layer.get_config()['name'])
        if f"{conv_name}block{str(i)}_0_conv" in layer.get_config()['name']:
            pytorch_block.downsample[0] = convert_conv(pytorch_block.downsample[0], layer)
        if f"{conv_name}block{str(i)}_0_bn" in layer.get_config()['name']:
            pytorch_block.downsample[1] = convert_bn(pytorch_block.downsample[1], layer)

        if f"{conv_name}block{str(i)}_1_conv" in layer.get_config()['name']:
            pytorch_block.conv1 = convert_conv(pytorch_block.conv1, layer)
        if f"{conv_name}block{str(i)}_1_bn" in layer.get_config()['name']:
            pytorch_block.bn1 = convert_bn(pytorch_block.bn1, layer)
        if f"{conv_name}block{str(i)}_2_conv" in layer.get_config()['name']:
            pytorch_block.conv2 = convert_conv(pytorch_block.conv2, layer)
        if f"{conv_name}block{str(i)}_2_bn" in layer.get_config()['name']:
            pytorch_block.bn2 = convert_bn(pytorch_block.bn2, layer)
        if f"{conv_name}block{str(i)}_3_conv" in layer.get_config()['name']:
            pytorch_block.conv3 = convert_conv(pytorch_block.conv3, layer)
        if f"{conv_name}block{str(i)}_3_bn" in layer.get_config()['name']:
            pytorch_block.bn3 = convert_bn(pytorch_block.bn3, layer)
#
# pytorch_model.layer1[0].get_submodule("bn1")
#
# pytorch_model.conv1.weight.data = torch.tensor(np.transpose(tf_keras_model.get_layer("conv1_conv").kernel.numpy(), (3, 2, 0, 1)))
# pytorch_model.conv1.bias.data = torch.tensor(tf_keras_model.get_layer("conv1_conv").bias.numpy())
#
# pytorch_model.bn1.weight.data = torch.tensor(tf_keras_model.get_layer("conv1_bn").gamma.numpy())
# pytorch_model.bn1.bias.data = torch.tensor(tf_keras_model.get_layer("conv1_bn").beta.numpy())
# pytorch_model.bn1.running_mean.data = torch.tensor(tf_keras_model.get_layer("conv1_bn").moving_mean.numpy())
# pytorch_model.bn1.running_var.data = torch.tensor(tf_keras_model.get_layer("conv1_bn").moving_variance.numpy())

torch.set_printoptions(precision=10)
with torch.no_grad():
    x_pt = torch.ones(1,3,224,224)
    outputs_pt = pytorch_model.conv1(x_pt)
    outputs_pt = pytorch_model.bn1(outputs_pt)

x_tf = tf.ones((1,224,224, 3))
outputs_tf = tf_keras_model.get_layer("conv1_pad")(x_tf)
outputs_tf = tf_keras_model.get_layer("conv1_conv")(outputs_tf)
outputs_tf = tf_keras_model.get_layer("conv1_bn")(outputs_tf)
outputs_tf = np.transpose(outputs_tf.numpy(), (0, 3, 1, 2))

np.allclose(outputs_tf, outputs_pt.numpy(), atol=1e-06)
