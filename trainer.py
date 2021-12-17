import tensorflow as tf
from metrics import MaskedSoftmaxCELoss
from moduls.model import EncoderDecoder
from tensorflow.keras.optimizers import Adam
from loader import DatasetLoader
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model


def encode(input_shape):
    backbone = MobileNetV2(include_top=True, weights="imagenet", input_shape=input_shape)
    for layer in backbone.layers:
        layer.trainable = False
    return Model(backbone.input, backbone.layers[-2].output)  # (1, 1280)


class trainer:
    def __init__(self, input_shape=(224, 224, 3),
                 seq_length=30,
                 lr=1e-4,
                 batch_size=32,
                 epochs=5,
                 meta_dir="dataset/Flicker8k_Dataset",
                 meta_file="dataset/Flickr8k.token.txt",
                 checkpoints="model.h5"
                 # meta_dir="/content/drive/MyDrive/Flickr8k/Imgs/",
                 # meta_file="/content/drive/MyDrive/Flickr8k/text/Flickr8k.token.txt"
                 ):
        self.seq_length = seq_length
        self.epochs = epochs
        self.batch_size = batch_size

        self.checkpoints = checkpoints
        self.meta_dir = meta_dir
        self.meta_file = meta_file

        self.loader = DatasetLoader(self.meta_dir, self.meta_file,
                                    batch_size=batch_size,
                                    maxlen=seq_length,
                                    size_image=input_shape[0])
        self.tokenizer = self.loader.load_tokenizer()
        self.vocal_size = len(self.tokenizer.word_docs) + 1

        self.encode_model = encode(input_shape)
        d_model = self.encode_model.output_shape[-1]

        self.optimizer = Adam(learning_rate=lr)
        self.model = EncoderDecoder(d_model, self.vocal_size)

    def train_step(self, x, target):
        # Encode
        # Encode with CV
        sos = tf.reshape(tf.constant([self.tokenizer.word_index['<sos>']] * len(target)),
                         shape=(-1, 1))
        dec_input = tf.concat([sos, target[:, :-1]], 1)

        x = preprocess_input(x)
        state = self.encode_model(x)
        with tf.GradientTape() as tape:
            pred, state = self.model(state, dec_input)
            loss = MaskedSoftmaxCELoss(target, pred)

        train_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(gradients, train_vars))
        return loss

    def validate_step(self, x, target):
        input_decode = ["<eos>"] * self.seq_length
        sentences = []
        # Gen words
        for _ in range(len(target)):
            prediction = self.model(x, input_decode, training=False)
            output = tf.argmax(prediction, axis=2).numpy()
            input_decode = output
            sentences.append(output)
        # Compare sentence with predictions and targets

    def train(self):
        sequences = self.loader.build_capture_loader()
        images = self.loader.build_image_loader()
        for epoch in range(self.epochs):
            for img, sequence in zip(images, sequences):
                loss = self.train_step(img, sequence)
                print(f"Epoch: {epoch} -- Loss: {loss}")
        self.model.save_weights(self.checkpoints)


if __name__ == '__main__':
    trainer().train()
