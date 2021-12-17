import tensorflow as tf
from tensorflow.keras.layers import GRU, Embedding, Dense
from tensorflow.keras.models import Model


class EncoderDecoder(Model):
    def __init__(self, d_model, vocal_size=1000):
        super(EncoderDecoder, self).__init__()

        # Decode
        self.emb_decode = Embedding(vocal_size, d_model)
        self.decode_layer_1 = GRU(d_model,
                                  return_state=True,
                                  return_sequences=True)
        self.decode_layer_2 = Dense(vocal_size)

    def __call__(self, state, target, training=True):
        # Decode
        decode = self.emb_decode(target, training=training)
        decode, state = self.decode_layer_1(decode, state, training=training)
        decode = self.decode_layer_2(decode, training=training)
        return decode, state


if __name__ == '__main__':
    sample_encoder = EncoderDecoder(1280)
    temp_input = tf.random.uniform((32, 1280), dtype=tf.float32, minval=0., maxval=1.)
    temp_target = tf.random.uniform((32, 30), dtype=tf.float32, minval=0, maxval=200)
    output = sample_encoder(temp_input, temp_target)
    print(output[0])
    # print(output)
