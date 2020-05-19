import argparse
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam


def add_common_layers(y):
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)

    return y


def grouped_convolution(y, nb_channels, _strides):
    return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)


def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
    """
    Our network consists of a stack of residual blocks. These blocks have the same topology,
    and are subject to two simple rules:
    - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
    - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
    """

    shortcut = y

    # ResNet (identical to ResNet when `cardinality` == 1)
    y = grouped_convolution(y, nb_channels_in, _strides=_strides)
    y = add_common_layers(y)

    y = grouped_convolution(y, nb_channels_in, _strides=_strides)

    # batch normalization is employed after aggregating the transformations and before adding to the shortcut
    y = layers.BatchNormalization()(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    y = layers.add([shortcut, y])

    # relu is performed right after each batch normalization,
    # expect for the output of the block where relu is performed after the adding to the shortcut
    y = layers.LeakyReLU()(y)

    return y

# args = {
#     'filter_num': 64,
#     'epochs': 1,
#     'dropout': 0.3,
#     'lr': 0.001,
#     'batch_size': 32,
#     'residualblock_num': 1,
#     'input_shape': (15, 15, 2),
#     'num_actions': 15 * 15,
# }


def main(args):
    # initialization of residual neural network
    # input layers
    input_tensor = layers.Input(shape=args.input_shape)

    # starting convolution layers
    x = layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
    x = add_common_layers(x)

    # residual blocks
    for i in range(args.residual_blocks):
        # project_shortcut = True if i == 0 else False
        x = residual_block(x, 256, 256)

    # policy head
    pi = layers.Conv2D(2, kernel_size=(1, 1), padding='same')(x)
    pi = layers.BatchNormalization()(pi)
    pi = layers.LeakyReLU()(pi)
    pi = layers.Flatten()(pi)
    pi = layers.Dense(args.num_actions, activation="softmax", name='pi')(pi)

    # value head
    v = layers.Conv2D(1, kernel_size=(1, 1), padding='same')(x)
    v = layers.BatchNormalization()(v)
    v = layers.LeakyReLU()(v)
    v = layers.Flatten()(v)
    v = layers.Dense(256)(v)
    v = layers.LeakyReLU()(v)
    v = layers.Dense(1, activation="tanh", name='v')(v)

    model = models.Model(inputs=[input_tensor], outputs=[pi, v])
    model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(args.lr))

    model.save(args.output, save_format='h5')
    print(model.summary())


if __name__ == '__main__':
    # example:
    # python make_network.py --width 5 --height 5 --residual_blocks 2 --lr 0.01 --dropout 0.3 --output model.h5
    # TODO add help
    parser = argparse.ArgumentParser()
    parser.add_argument('--grounding', action='store_true')
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    parser.add_argument('--residual_blocks', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--output', type=str)

    args = parser.parse_args()

    args.__setattr__('input_shape', (args.width, args.height, 2))
    args.__setattr__('num_actions', args.width * args.height + int(args.grounding))

    main(args)