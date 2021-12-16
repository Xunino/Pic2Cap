import tensorflow as tf
from tensorflow.keras.layers import GRU, Embedding
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model


class EncoderDecoder(Model):
    def __init__(self, input_shape=(224, 224, 3), vocal_size=1000):
        super(EncoderDecoder, self).__init__()

        # Encode with CV
        self.backbone = MobileNetV2(include_top=True, weights="imagenet", input_shape=input_shape)
        for layer in self.backbone.layers:
            layer.trainable = False
        self.encode_model = Model(self.backbone.input, self.backbone.layers[-2].output)  # (1, 1280)
        d_model = self.encode_model.output_shape[-1]

        # Decode
        self.emb_decode = Embedding(vocal_size, d_model)
        self.decode_layer_1 = GRU(d_model, return_sequences=True)
        self.decode_layer_2 = GRU(vocal_size)

    def __call__(self, input, target, training=True):
        # Encode
        input = preprocess_input(input)
        state = self.encode_model(input, training=True)

        # Decode
        decode = self.emb_decode(target, training=True)
        decode = self.decode_layer_1(decode, state, training=True)
        decode = self.decode_layer_2(decode, training=True)
        return decode


if __name__ == '__main__':
    sample_encoder = EncoderDecoder()
    temp_input = tf.random.uniform((1, 224, 224, 3), dtype=tf.float32, minval=0, maxval=300)
    temp_target = tf.random.uniform((1, 10), dtype=tf.float32, minval=0, maxval=200)
    output = sample_encoder(temp_input, temp_target)
    print(output.shape)
    print(output)
