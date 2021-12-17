import tensorflow as tf
from tensorflow.keras.layers import GRU, Embedding, Dense
from tensorflow.keras.models import Model


class EncoderDecoder(Model):
    def __init__(self, vocal_size, d_model=128, embedding_size=64):
        super(EncoderDecoder, self).__init__()
        self.dense = Dense(d_model)

        # Decode
        self.emb_decode = Embedding(vocal_size, embedding_size)
        self.decode_layer_1 = GRU(d_model,
                                  return_state=True,
                                  return_sequences=True)
        self.decode_layer_2 = Dense(vocal_size)

    def __call__(self, encode_out, decode_input, training=True):
        encode_out = self.dense(encode_out)

        # Decode
        decode = self.emb_decode(decode_input, training=training)
        decode, state = self.decode_layer_1(decode, encode_out, training=training)
        decode = self.decode_layer_2(decode, training=training)
        return decode, state


if __name__ == '__main__':
    sample_encoder = EncoderDecoder(5000, 2000)
    temp_input = tf.random.uniform((1, 1280), dtype=tf.float32, minval=0., maxval=1.)
    temp_target = tf.random.uniform((1, 1), dtype=tf.float32, minval=0, maxval=200)
    output = sample_encoder(temp_input, temp_target)
    print(output[0])
    # print(output)
