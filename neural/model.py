class RNN:
    def __init__(self):
        # initialization of residual neural network
        # input layers
        input_tensor = layers.Input(shape=args.input_shape)

        # starting convolution layers
        x = layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        x = self.add_common_layers(x)

        # residual blocks
        for i in range(args.residualblock_num):
            # project_shortcut = True if i == 0 else False
            x = self.residual_block(x, 256, 256)

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

        self.model = models.Model(inputs=[input_tensor], outputs=[pi, v])
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(args.lr))

        print(self.model.summary())

    @classmethod
    def from_file(cls, filename):
        inst = RNN.__new__(RNN)
        inst.model = models.load_model(filename)

        return inst

    def add_common_layers(self, y):
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)

        return y

    def grouped_convolution(self, y, nb_channels, _strides):
        return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)

    def residual_block(self, y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
        """
        Our network consists of a stack of residual blocks. These blocks have the same topology,
        and are subject to two simple rules:
        - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
        - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
        """

        shortcut = y

        # ResNet (identical to ResNet when `cardinality` == 1)
        y = self.grouped_convolution(y, nb_channels_in, _strides=_strides)
        y = self.add_common_layers(y)

        y = self.grouped_convolution(y, nb_channels_in, _strides=_strides)

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

    def train(self, examples):
        inputs_field, true_pi, true_v = list(map(np.asarray, zip(*examples)))
        self.model.fit(
            x=inputs_field,
            y=[true_pi, true_v],
            batch_size=args.batch_size,
            epochs=1,
            # callbacks=[self.checkpoint, self.tbCallback]
        )

    def predict(self, field):
        field = np.reshape(field, (-1, *field.shape))
        pi, v = self.model.predict(field)
        return pi[0], v[0]

    def save(self, filename):
        self.model.save(filename, save_format='h5')
