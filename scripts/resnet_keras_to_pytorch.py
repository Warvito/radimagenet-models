"""
Script to transform the weights of the ResNetmodel from Keras to Pytorch.

Script based on https://github.com/BMEII-AI/RadImageNet/issues/3#issuecomment-1232417600
and https://discuss.pytorch.org/t/transferring-weights-from-keras-to-pytorch/9889
"""
import argparse

import numpy as np
import tensorflow as tf
import torch

from radimagenet_models.models.resnet import ResNet50

torch.set_printoptions(precision=10)

input_path = "/media/walter/Storage/Projects/radimagenet-models/outputs/original_weights/RadImageNet-ResNet50_notop.h5"
out_path = "/media/walter/Storage/Projects/radimagenet-models/outputs/pytorch_weights/RadImageNet-ResNet50_notop.pth"


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


def convert_stack(pytorch_stack, keras_model, stack_name, num_blocks):
    layers_list = []
    for layer in keras_model.layers:
        if stack_name in layer.get_config()["name"]:
            layers_list.append(layer)

    for i in range(1, num_blocks + 1):
        pytorch_block = pytorch_stack[i - 1]
        for layer in layers_list:
            if f"{stack_name}_block{str(i)}_0_conv" in layer.get_config()["name"]:
                pytorch_block.downsample[0] = convert_conv(pytorch_block.downsample[0], layer)
            elif f"{stack_name}_block{str(i)}_0_bn" in layer.get_config()["name"]:
                pytorch_block.downsample[1] = convert_bn(pytorch_block.downsample[1], layer)
            elif f"{stack_name}_block{str(i)}_1_conv" in layer.get_config()["name"]:
                pytorch_block.conv1 = convert_conv(pytorch_block.conv1, layer)
            elif f"{stack_name}_block{str(i)}_1_bn" in layer.get_config()["name"]:
                pytorch_block.bn1 = convert_bn(pytorch_block.bn1, layer)
            elif f"{stack_name}_block{str(i)}_2_conv" in layer.get_config()["name"]:
                pytorch_block.conv2 = convert_conv(pytorch_block.conv2, layer)
            elif f"{stack_name}_block{str(i)}_2_bn" in layer.get_config()["name"]:
                pytorch_block.bn2 = convert_bn(pytorch_block.bn2, layer)
            elif f"{stack_name}_block{str(i)}_3_conv" in layer.get_config()["name"]:
                pytorch_block.conv3 = convert_conv(pytorch_block.conv3, layer)
            elif f"{stack_name}_block{str(i)}_3_bn" in layer.get_config()["name"]:
                pytorch_block.bn3 = convert_bn(pytorch_block.bn3, layer)

        pytorch_stack[i - 1] = pytorch_block
    return pytorch_stack


def main(args):
    pytorch_model = ResNet50()
    keras_model = tf.keras.models.load_model(args.input_path)

    # Convert weights
    pytorch_model.conv1 = convert_conv(pytorch_model.conv1, keras_model.get_layer("conv1_conv"))
    pytorch_model.bn1 = convert_bn(pytorch_model.bn1, keras_model.get_layer("conv1_bn"))

    pytorch_model.layer1 = convert_stack(pytorch_model.layer1, keras_model, "conv2", num_blocks=3)
    pytorch_model.layer2 = convert_stack(pytorch_model.layer2, keras_model, "conv3", num_blocks=4)
    pytorch_model.layer3 = convert_stack(pytorch_model.layer3, keras_model, "conv4", num_blocks=6)
    pytorch_model.layer4 = convert_stack(pytorch_model.layer4, keras_model, "conv5", num_blocks=3)

    # Test converted model
    x = np.random.rand(1, 224, 224, 3)
    x_pt = torch.from_numpy(np.transpose(x, (0, 3, 1, 2))).float()

    pytorch_model.eval()
    with torch.no_grad():
        outputs_pt = pytorch_model(x_pt)
        outputs_pt = np.transpose(outputs_pt.numpy(), (0, 2, 3, 1))

    x_tf = tf.convert_to_tensor(x)
    outputs_tf = keras_model(x_tf, training=False)
    outputs_tf = outputs_tf.numpy()

    print(f"Are the outputs all close (absolute tolerance = 1e-04)? {np.allclose(outputs_tf, outputs_pt, atol=1e-04)}")
    print("Pytorch output")
    print(outputs_pt[0, :30, 0, 0])
    print("Tensoflow Keras output")
    print(outputs_tf[0, :30, 0, 0])

    # Saving model
    torch.save(pytorch_model.state_dict(), args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Path to the original RadImageNet-ResNet50_notop.h5 file.")
    parser.add_argument("--output_path", help="Path to save the converted .pth file.")
    args = parser.parse_args()

    main(args)
