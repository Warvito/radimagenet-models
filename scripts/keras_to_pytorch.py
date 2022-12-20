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

def convert_stack(pytorch_stack, keras_model, stack_name, num_blocks):
    layers_list = []
    for layer in keras_model.layers:
        if stack_name in layer.get_config()['name']:
            layers_list.append(layer)
    print(f"we have {len(layers_list)} layers")

    list_of_added = []
    for i in range(1, num_blocks+1):
        pytorch_block = pytorch_stack[i-1]
        for layer in layers_list:
            # print(layer.name)
            if f"{stack_name}_block{str(i)}_0_conv" in layer.get_config()['name']:
                pytorch_block.downsample[0] = convert_conv(pytorch_block.downsample[0], layer)
                list_of_added.append(layer)
            elif f"{stack_name}_block{str(i)}_0_bn" in layer.get_config()['name']:
                pytorch_block.downsample[1] = convert_bn(pytorch_block.downsample[1], layer)
                list_of_added.append(layer)
            elif f"{stack_name}_block{str(i)}_1_conv" in layer.get_config()['name']:
                pytorch_block.conv1 = convert_conv(pytorch_block.conv1, layer)
                list_of_added.append(layer)
            elif f"{stack_name}_block{str(i)}_1_bn" in layer.get_config()['name']:
                pytorch_block.bn1 = convert_bn(pytorch_block.bn1, layer)
                list_of_added.append(layer)
            elif f"{stack_name}_block{str(i)}_2_conv" in layer.get_config()['name']:
                pytorch_block.conv2 = convert_conv(pytorch_block.conv2, layer)
                list_of_added.append(layer)
            elif f"{stack_name}_block{str(i)}_2_bn" in layer.get_config()['name']:
                pytorch_block.bn2 = convert_bn(pytorch_block.bn2, layer)
                list_of_added.append(layer)
            elif f"{stack_name}_block{str(i)}_3_conv" in layer.get_config()['name']:
                pytorch_block.conv3 = convert_conv(pytorch_block.conv3, layer)
                list_of_added.append(layer)
            elif f"{stack_name}_block{str(i)}_3_bn" in layer.get_config()['name']:
                pytorch_block.bn3 = convert_bn(pytorch_block.bn3, layer)
                list_of_added.append(layer)

        pytorch_stack[i - 1] = pytorch_block
    A = list(set(layers_list) - set(list_of_added))
    for i in A:
        print(i.name)
    return pytorch_stack


pytorch_model = ResNet50()
pytorch_model.eval()
keras_model = tf.keras.models.load_model(input_path)

pytorch_model.conv1 = convert_conv(pytorch_model.conv1, keras_model.get_layer("conv1_conv"))
pytorch_model.bn1 = convert_bn(pytorch_model.bn1, keras_model.get_layer("conv1_bn"))

pytorch_model.layer1 = convert_stack(pytorch_model.layer1, keras_model, "conv2", num_blocks=3)
pytorch_model.layer2 = convert_stack(pytorch_model.layer2, keras_model, "conv3", num_blocks=4)
pytorch_model.layer3 = convert_stack(pytorch_model.layer3, keras_model, "conv4", num_blocks=6)
pytorch_model.layer4 = convert_stack(pytorch_model.layer4, keras_model, "conv5", num_blocks=3)


torch.set_printoptions(precision=10)
pytorch_model.eval()
with torch.no_grad():
    x_pt = torch.ones(1,3,224,224)
    outputs_pt = pytorch_model(x_pt)
    outputs_pt = np.transpose(outputs_pt.numpy(), (0, 2, 3, 1))

x_tf = tf.ones((1, 224,224, 3))
outputs_tf = keras_model(x_tf, training=False)
outputs_tf = outputs_tf.numpy()
print(np.allclose(outputs_tf, outputs_pt, atol=1e-01))





torch.set_printoptions(precision=10)
pytorch_model.eval()
x = np.random.rand(1,224,224,3)

x_pt = torch.from_numpy(np.transpose(x,(0,3,1,2))).float()
# x_pt = torch.ones((1, 3, 224, 224))
with torch.no_grad():
    h = pytorch_model.conv1(x_pt)
    h = pytorch_model.bn1(h)
    h = pytorch_model.relu(h)
    h1 = pytorch_model.maxpool(h)

    h = pytorch_model.layer1[0].conv1(h1)
    h = pytorch_model.layer1[0].bn1(h)
    h = pytorch_model.layer1[0].relu(h)
    h = pytorch_model.layer1[0].conv2(h)
    h = pytorch_model.layer1[0].bn2(h)
    h = pytorch_model.layer1[0].relu(h)
    h = pytorch_model.layer1[0].conv3(h)
    h = pytorch_model.layer1[0].bn3(h)
    h1 = pytorch_model.layer1[0].downsample(h1)
    h = h + h1
    h2 = pytorch_model.layer1[0].relu(h)
    h=h2

    h = pytorch_model.layer1[1].conv1(h)
    h = pytorch_model.layer1[1].bn1(h)
    h = pytorch_model.layer1[1].relu(h)
    h = pytorch_model.layer1[1].conv2(h)
    h = pytorch_model.layer1[1].bn2(h)
    h = pytorch_model.layer1[1].relu(h)
    h = pytorch_model.layer1[1].conv3(h)
    h = pytorch_model.layer1[1].bn3(h)
    h = h + h2
    h3 = pytorch_model.layer1[1].relu(h)
    h=h3

    h = pytorch_model.layer1[2].conv1(h)
    h = pytorch_model.layer1[2].bn1(h)
    h = pytorch_model.layer1[2].relu(h)
    h = pytorch_model.layer1[2].conv2(h)
    h = pytorch_model.layer1[2].bn2(h)
    h = pytorch_model.layer1[2].relu(h)
    h = pytorch_model.layer1[2].conv3(h)
    h = pytorch_model.layer1[2].bn3(h)
    h = h + h3
    h4 = pytorch_model.layer1[2].relu(h)
    h=h4

    # h = pytorch_model.layer2[0].conv1(h)
    # h = pytorch_model.layer2[0].bn1(h)
    # h = pytorch_model.layer2[0].relu(h)
    # h = pytorch_model.layer2[0].conv2(h)
    # h = pytorch_model.layer2[0].bn2(h)
    # h = pytorch_model.layer2[0].relu(h)
    # h = pytorch_model.layer2[0].conv3(h)
    # h = pytorch_model.layer2[0].bn3(h)
    # h4 = pytorch_model.layer2[0].downsample(h4)
    # h = h + h4
    # h2 = pytorch_model.layer2[0].relu(h)
    # h=h2

    out_pt = np.transpose(h.numpy(), (0, 2, 3, 1))


x_tf = tf.convert_to_tensor(x)
# x_tf = tf.ones((1, 224, 224, 3))
h = keras_model.get_layer("conv1_pad")(x_tf, training=False)
h = keras_model.get_layer("conv1_conv")(h, training=False)
h = keras_model.get_layer("conv1_bn")(h, training=False)
h = keras_model.get_layer("conv1_relu")(h, training=False)
h = keras_model.get_layer("pool1_pad")(h, training=False)
h1 = keras_model.get_layer("pool1_pool")(h, training=False)

h = keras_model.get_layer("conv2_block1_1_conv")(h1, training=False)
h = keras_model.get_layer("conv2_block1_1_bn")(h, training=False)
h = keras_model.get_layer("conv2_block1_1_relu")(h, training=False)
h = keras_model.get_layer("conv2_block1_2_conv")(h, training=False)
h = keras_model.get_layer("conv2_block1_2_bn")(h, training=False)
h = keras_model.get_layer("conv2_block1_2_relu")(h, training=False)
h = keras_model.get_layer("conv2_block1_3_conv")(h, training=False)
h = keras_model.get_layer("conv2_block1_3_bn")(h, training=False)
h1 = keras_model.get_layer("conv2_block1_0_conv")(h1, training=False)
h1 = keras_model.get_layer("conv2_block1_0_bn")(h1, training=False)
h = keras_model.get_layer("conv2_block1_add")([h1, h], training=False)
h2 = keras_model.get_layer("conv2_block1_out")(h, training=False)
h = h2

h = keras_model.get_layer("conv2_block2_1_conv")(h, training=False)
h = keras_model.get_layer("conv2_block2_1_bn")(h, training=False)
h = keras_model.get_layer("conv2_block2_1_relu")(h, training=False)
h = keras_model.get_layer("conv2_block2_2_conv")(h, training=False)
h = keras_model.get_layer("conv2_block2_2_bn")(h, training=False)
h = keras_model.get_layer("conv2_block2_2_relu")(h, training=False)
h = keras_model.get_layer("conv2_block2_3_conv")(h, training=False)
h = keras_model.get_layer("conv2_block2_3_bn")(h, training=False)
h = keras_model.get_layer("conv2_block2_add")([h2, h], training=False)
h3 = keras_model.get_layer("conv2_block2_out")(h, training=False)
h=h3

h = keras_model.get_layer("conv2_block3_1_conv")(h3, training=False)
h = keras_model.get_layer("conv2_block3_1_bn")(h, training=False)
h = keras_model.get_layer("conv2_block3_1_relu")(h, training=False)
h = keras_model.get_layer("conv2_block3_2_conv")(h, training=False)
h = keras_model.get_layer("conv2_block3_2_bn")(h, training=False)
h = keras_model.get_layer("conv2_block3_2_relu")(h, training=False)
h = keras_model.get_layer("conv2_block3_3_conv")(h, training=False)
h = keras_model.get_layer("conv2_block3_3_bn")(h, training=False)
h = keras_model.get_layer("conv2_block3_add")([h3, h], training=False)
h4 = keras_model.get_layer("conv2_block3_out")(h, training=False)
h=h4

# h = keras_model.get_layer("conv3_block1_1_conv")(h, training=False)
# h = keras_model.get_layer("conv3_block1_1_bn")(h, training=False)
# h = keras_model.get_layer("conv3_block1_1_relu")(h, training=False)
# h = keras_model.get_layer("conv3_block1_2_conv")(h, training=False)
# h = keras_model.get_layer("conv3_block1_2_bn")(h, training=False)
# h = keras_model.get_layer("conv3_block1_2_relu")(h, training=False)
# h = keras_model.get_layer("conv3_block1_3_conv")(h, training=False)
# h = keras_model.get_layer("conv3_block1_3_bn")(h, training=False)
# h4 = keras_model.get_layer("conv3_block1_0_conv")(h4, training=False)
# h4 = keras_model.get_layer("conv3_block1_0_bn")(h4, training=False)
# h = keras_model.get_layer("conv3_block1_add")([h4, h], training=False)
# h2 = keras_model.get_layer("conv3_block1_out")(h, training=False)
# h = h2

out_tf = h.numpy()

print(np.allclose(out_tf, out_pt, atol=1e-04))

print(out_pt[0,:30,0,1])
print(out_tf[0,:30,0,1])
