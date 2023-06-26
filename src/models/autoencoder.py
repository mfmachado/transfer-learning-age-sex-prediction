from keras import layers
from tensorflow.keras.models import Model


class AutoEncoder:

    def create(self, layers_number: list):
        if len(layers_number) == 0:
            raise AttributeError("The layers list is empty")

        input_layer = layers.Input(shape=(121, 145, 121, 1))

        # Encoder
        encoder = layers.Cropping3D(cropping=((0, 0), (8, 9), (0, 0)))(input_layer)
        encoder = layers.ZeroPadding3D(padding=((3, 4), (0, 0), (3, 4)))(encoder)

        # Encoder
        for num_filters_layer in layers_number:
            encoder = layers.Conv3D(num_filters_layer, 3, strides=1, activation=None, padding="same")(encoder)
            encoder = layers.BatchNormalization()(encoder)
            encoder = layers.MaxPool3D(pool_size=(2, 2, 2))(encoder)
            encoder = layers.ReLU()(encoder)

        # Decoder
        layers_decode = layers_number.copy()
        layers_decode.reverse()
        decoder = None
        for num_filters_layer in layers_decode:
            if decoder is None:
                decoder = layers.Conv3D(num_filters_layer, 3, activation=None, strides=1,
                                        padding="same")(encoder)
            else:
                decoder = layers.Conv3D(num_filters_layer, 3, activation=None, strides=1,
                                        padding="same")(decoder)

            decoder = layers.BatchNormalization()(decoder)
            decoder = layers.UpSampling3D(size=(2, 2, 2))(decoder)
            decoder = layers.ReLU()(decoder)

        decoder = layers.Conv3D(1, 3, activation="sigmoid", padding="same")(decoder)
        decoder = layers.ZeroPadding3D(padding=((0, 0), (8, 9), (0, 0)))(decoder)

        decoder = layers.Cropping3D(cropping=((3, 4), (0, 0), (3, 4)))(decoder)

        # Autoencoder
        autoencoder_model = Model(input_layer, decoder)
        autoencoder_model.compile(optimizer="adam", loss="mse")
        return autoencoder_model
