import tensorflow as tf
from metrics import MaskSoftmaxCELoss
from moduls.model import EncoderDecoder
from tensorflow.keras.optimizers import Adam
from loader import DatasetLoader


class trainer:
    def __init__(self, input_shape=(224, 224, 3),
                 seq_length=30,
                 lr=1e-4,
                 batch_size=32,
                 meta_dir="dataset/Flicker8k_Dataset/",
                 meta_file="dataset/Flickr8k.token.txt"):
        self.seq_length = seq_length

        self.meta_dir = meta_dir
        self.meta_file = meta_file

        self.loader = DatasetLoader(self.meta_dir, self.meta_file,
                                    batch_size=batch_size,
                                    maxlen=seq_length,
                                    size_image=input_shape[0])
        self.tokenizer = self.loader.load_tokenizer()
        self.vocal_size = len(self.tokenizer.word_docs) + 1

        self.optimizer = Adam(learning_rate=lr)
        self.model = EncoderDecoder(input_shape, self.vocal_size)

    def train_step(self, x, target):
        with tf.GradientTape() as tape:
            x = self.model(x, target)
            loss = MaskSoftmaxCELoss(target, x)

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
        for img, sequence in zip(images, sequences):
            print(img.shape, sequence.shape)
            loss = self.train_step(img, sequence)
            print(loss)


if __name__ == '__main__':
    trainer(batch_size=1).train()
