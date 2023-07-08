from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, DepthwiseConv2D
from tensorflow.keras.models import Model
from IPython import embed
import tensorflow as tf
from tensorflow.keras import layers
import keras

expansion = 1
BN_MOMENTUM = 0.1

def conv_bn_relu(inputs, planes):
    x = tf.keras.layers.Conv2D(planes, kernel_size=1, use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization(momentum=BN_MOMENTUM)(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def depthwiseconv_bn_relu(inputs, stride):
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization(momentum=BN_MOMENTUM)(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def AddRelu3(inputs):
    added = tf.keras.layers.Add()(inputs)
    output = tf.keras.layers.ReLU()(added)
    return output


def LW_Bottleneck(inputs, inplanes, planes, stride=1, downsample=None, attention='GC'):

    out = conv_bn_relu(inputs, planes)
    out = depthwiseconv_bn_relu(out, stride)

    out = tf.keras.layers.Conv2D(int(planes*expansion), kernel_size=1, use_bias =False)(out)
    out = tf.keras.layers.BatchNormalization(momentum = BN_MOMENTUM)(out)

    if attention == 'SE':
        out = SELayer(planes * expansion, out)
    elif attention == 'GC':
        out_planes = planes * expansion //16 if planes* expansion//16 >=16 else 16
        out = GCBlock(out, planes * expansion, out_planes, 'att', ['channel_add'])
    else:
        out = 0
    
    if downsample is not None:
        inputs = tf.keras.layers.Conv2D(int(planes*expansion), kernel_size=1, strides = stride, use_bias =False)(inputs)
        inputs = tf.keras.layers.BatchNormalization(momentum = BN_MOMENTUM)(inputs)

    x = Add()([inputs, out])
    x = ReLU()(x)

    return x



def GCBlock(x, inplanes, planes, pool, fusions):
    assert pool in ['avg', 'att']
    assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
    assert len(fusions) > 0, 'at least one fusion should be used'
    batch, height, width, channel = x.shape.as_list()

    if 'att' in pool:
        input_x = x
        conv_mask = tf.keras.layers.Conv2D(1, kernel_size=1)(x)
        conv_mask = tf.keras.layers.Softmax(axis=2)(conv_mask)
    else:
        conv_mask = tf.keras.layers.GlobalAvgPool2D()

    if 'channel_add' in fusions:
        channel_add_conv = tf.keras.layers.Conv2D(planes, kernel_size=1)(conv_mask)
        channel_add_conv = tf.keras.layers.LayerNormalization(axis = -1)(channel_add_conv)
        channel_add_conv = tf.keras.layers.ReLU()(channel_add_conv)
        channel_add_conv = tf.keras.layers.Conv2D(inplanes, kernel_size=1)(channel_add_conv)
    else:
        channel_add_conv = None

    if 'channel_mul' in fusions:
        channel_mul_conv = tf.keras.layers.Conv2D(planes, kernel_size=1)(conv_mask)
        channel_mul_conv = tf.keras.layers.LayerNormalization(axis = -1)(channel_mul_conv)
        channel_mul_conv = tf.keras.layers.ReLU()(channel_mul_conv)
        channel_mul_conv = tf.keras.layers.Conv2D(inplanes, kernel_size=1)(channel_mul_conv)
    else:
        channel_mul_conv = None

    if channel_mul_conv is not None:
        channel_mul_term = tf.keras.layers.Activaation(activation = 'sigmoid')(channel_mul_conv)
        out = tf.keras.layers.Multiply()([x, channel_mul_term])
    else:
        out = x

    if channel_add_conv is not None:
       channel_add_term =  channel_add_conv
       out = tf.keras.layers.Add()([out, channel_add_term])

    return out 


def LPN(x, layer, cfg):
    extra = cfg['MODEL']['EXTRA']
    inplanes = 64
    deconv_with_bias = extra['DECONV_WITH_BIAS']
    attention = extra.get('ATTENTION')

    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(momentum=BN_MOMENTUM)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    x = _make_layer(x, 64, layer[0], inplanes, attention=attention)
    x = _make_layer(x, 128, layer[1],inplanes, stride=2, attention=attention)
    x = _make_layer(x, 256, layer[2],inplanes, stride=2, attention=attention)
    x = _make_layer(x, 512, layer[3],inplanes, stride=1, attention=attention)

    # used for deconv layers
    x = _make_deconv_layer(x, extra['NUM_DECONV_LAYERS'], extra['NUM_DECONV_FILTERS'], extra['NUM_DECONV_KERNELS'], deconv_with_bias)

    final_layer = tf.keras.layers.Conv2D(cfg["MODEL"]["NUM_JOINTS"], kernel_size=extra['FINAL_CONV_KERNEL'], strides=1,padding='same' if extra['FINAL_CONV_KERNEL'] == 3 else 'valid')(x)
    return final_layer

def _make_layer(x, planes, blocks,inplanes, stride=1, attention=None):
    downsample = None
    if stride != 1 or inplanes != planes * expansion:
        downsample = "Not None"
    layers_ = LW_Bottleneck(x, inplanes, planes=planes, stride=stride, downsample=downsample, attention=attention)
    for i in range(1, blocks):
        layers_ = LW_Bottleneck(layers_,inplanes, planes=planes, stride=1, attention=attention)

    return layers_

def _make_deconv_layer(input_, num_layers, num_filters, num_kernels, deconv_with_bias):
    x = input_
    for i in range(num_layers):
        kernel, padding, output_padding = _get_deconv_cfg(num_kernels[i], i)
        planes = num_filters[i]
        x = layers.Conv2DTranspose(planes, kernel_size=kernel, strides=2, padding='same', output_padding=output_padding, groups=planes, use_bias=deconv_with_bias)(x)
        x = layers.BatchNormalization(momentum=BN_MOMENTUM)(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(planes, kernel_size=1, use_bias=False)(x)
        x = layers.BatchNormalization(momentum=BN_MOMENTUM)(x)
        x = layers.ReLU()(x)

        inplanes = planes
    return x

def _get_deconv_cfg(deconv_kernel, index):
    if deconv_kernel == 4:
        padding = 'same'
        output_padding = 1
    elif deconv_kernel == 3:
        padding = 'same'
        output_padding = 1
    elif deconv_kernel == 2:
        padding = 'valid'
        output_padding = 0
    return deconv_kernel, padding, output_padding



# used for deconv layers
def get_deconv_cfg(deconv_kernel, index):
    if deconv_kernel == 4:
        padding = 1
        output_padding = 0
    elif deconv_kernel == 3:
        padding = 1
        output_padding = 1
    elif deconv_kernel == 2:
        padding = 0
        output_padding = 0
    return deconv_kernel, padding, output_padding

def make_deconv_layer(num_layers, num_filters, num_kernels):
    layers = []
    for i in range(num_layers):
        kernel, padding, output_padding = get_deconv_cfg(num_kernels[i], i)

        planes = num_filters[i]
        layers.append(tf.keras.layers.Conv2DTranspose(
            planes, kernel_size=kernel, strides=2, padding='same',
            output_padding=output_padding, use_bias=deconv_with_bias,
            groups=tf.gcd(inplanes, planes))(input))
        layers.append(bn(planes))
        layers.append(relu(planes))
        layers.append(tf.keras.layers.Conv2D(
            planes, kernel_size=1, use_bias=False)(planes))
        layers.append(bn(planes))
        layers.append(relu(planes))
        inplanes = planes

    return tf.keras.Sequential(layers)


resnet_spec = {
    50: (LW_Bottleneck, [3, 4, 6, 3]),
    101: (LW_Bottleneck, [3, 4, 23, 3]),
    152: (LW_Bottleneck, [3, 8, 36, 3])
}


def get_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg["MODEL"]["EXTRA"]["NUM_LAYERS"]
    imgsize= (cfg["img_height"],cfg["img_width"],3)
    block_class, layers = resnet_spec[num_layers]
    x = tf.keras.Input(shape= imgsize, name = 'img_input')
    lpn_out = LPN(x, layers, cfg, **kwargs)
    NUM_KEYPOINTS = cfg["MODEL"]["NUM_JOINTS"]
    outputs = tf.keras.layers.SeparableConv2D(NUM_KEYPOINTS, kernel_size=5, strides=1, activation="relu")(lpn_out)
    outputs = tf.keras.layers.SeparableConv2D(NUM_KEYPOINTS, kernel_size=3, strides=1, activation="sigmoid")(outputs)
    model  = tf.keras.models.Model(inputs = x, outputs = outputs)
    return model




