import tensorflow as tf
from tqdm import tqdm

from metrics import MaskedSoftmaxCELoss, CustomSchedule
from moduls.model import EncoderDecoder
from tensorflow.keras.optimizers import Adam
from loader import DatasetLoader
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model


def encode(input_shape):
    backbone = InceptionV3(include_top=True, weights="imagenet", input_shape=input_shape)
    for layer in backbone.layers:
        layer.trainable = False
    return Model(backbone.input, backbone.layers[-2].output)  # (1, 1280)


class trainer:
    def __init__(self, input_shape=(299, 299, 3),
                 seq_length=30,
                 lr=1e-3,
                 batch_size=32,
                 epochs=20,
                 #  meta_dir="dataset/Flicker8k_Dataset",
                 #  meta_file="dataset/Flickr8k.token.txt",
                 checkpoints="model.h5",
                 meta_dir="/content/drive/MyDrive/Flickr8k/Imgs/",
                 meta_file="/content/drive/MyDrive/Flickr8k/text/Flickr8k.token.txt"
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

        # Initialize learning rate scheduler
        learning_scheduler = CustomSchedule(d_model, batch_size // 2)
        self.optimizer = Adam(learning_rate=learning_scheduler)
        self.model = EncoderDecoder(self.vocal_size, d_model=d_model)

    def train_step(self, x, y):
        # Encode
        # Encode with CV
        in_decode = y[:, :-1]
        out_decode = y[:, 1:]

        x = preprocess_input(x)
        state = self.encode_model(x)
        with tf.GradientTape() as tape:
            pred, state = self.model(state, in_decode)
            loss = MaskedSoftmaxCELoss(out_decode, pred)

        train_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(gradients, train_vars))
        return loss

    def validate_step(self, xs, targets):
        for i in range(len(xs)):
            x = tf.expand_dims(xs[i], axis=0)
            target = tf.expand_dims(targets[i], axis=0)
            x = preprocess_input(x)
            state = self.encode_model(x)

            start = [self.tokenizer.word_index["<sos>"]]
            dec_input = tf.convert_to_tensor(start, dtype=tf.int64)
            dec_input = tf.expand_dims(dec_input, 0)

            sequences = []
            # Gen words
            for _ in range(len(target[-1])):
                prediction, state = self.model(state, dec_input, training=False)
                output = tf.argmax(prediction, axis=2).numpy()
                dec_input = output
                if output[0][0] == self.tokenizer.texts_to_sequences(["<eos>"]):
                    break
                sequences.append(output[0][0])

            # Compare sentence with predictions and targets
            print("\n[TARGET]   : ", self.tokenizer.sequences_to_texts(target.numpy())[0])
            print("[PREDICTED]: ", self.tokenizer.sequences_to_texts([sequences])[0])
            print("-" * 100 + "\n")

    def train(self):
        is_train = True
        sequences = self.loader.build_capture_loader()
        images = self.loader.build_image_loader()
        print(f"\nLen_images: {len(images)} -- Len_sequences: {len(sequences)}")

        for epoch in range(self.epochs):
            pbar = tqdm(enumerate(zip(images, sequences)), total=len(sequences)) if is_train else enumerate(
                zip(images, sequences))
            for it, (img, sequence) in pbar:
                loss = self.train_step(img, sequence)
                pbar.set_description(f"Epoch: {epoch} -- Loss: {loss}")
                if it % 200 == 0:
                    self.validate_step(img[-3:-1], sequence[-3:-1])

        # self.model.save_weights(self.checkpoints)


if __name__ == '__main__':
    batch_size = 128
    trainer(batch_size=batch_size).train()
