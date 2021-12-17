import os
import io
import json
import pathlib
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences


class DatasetLoader:
    def __init__(self, meta_dir, meta_file, batch_size=32, maxlen=120, size_image=224):
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.size_image = size_image
        self.meta_dir = meta_dir
        self.meta_file = meta_file
        self.saved_tokenizer = "./tokenizer.json"

        self.image_paths = []
        self.cap_images = []

        self.autotune = tf.data.experimental.AUTOTUNE

    def save_tokenizer(self, tokenizer):
        print(f"Saved tokenizer in {self.saved_tokenizer}")
        tokenizer_json = tokenizer.to_json()
        with io.open(self.saved_tokenizer, 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))
        f.close()

    def load_tokenizer(self):
        with io.open(self.saved_tokenizer, "r") as f:
            data = json.load(f)
            tokenizer = tokenizer_from_json(data)
        return tokenizer

    def loader(self):
        meta_data = open(self.meta_file, "r")
        for line in meta_data.readlines()[:1000]:
            l = line.strip().split("\t")
            path = os.path.join(self.meta_dir, l[0].split(".")[0] + ".jpg")
            if os.path.exists(path):
                self.image_paths.append(path)
                self.cap_images.append("<sos> " + l[-1] + " <eos>")
        meta_data.close()

    def build_capture_loader(self):
        self.loader()
        if not os.path.exists(self.saved_tokenizer):
            tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
            tokenizer.fit_on_texts(self.cap_images)
            self.save_tokenizer(tokenizer)
        else:
            tokenizer = self.load_tokenizer()

        sequences = tokenizer.texts_to_sequences(self.cap_images)
        sequences_padded = pad_sequences(sequences, maxlen=self.maxlen, padding="post", truncating="post")

        # Performance
        return self.config_for_text_performance(sequences_padded)

    def config_for_text_performance(self, ds):
        ds = tf.data.Dataset.from_tensor_slices(ds)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=self.autotune)
        return ds

    @staticmethod
    def processing_image(file_image):
        img = tf.io.read_file(file_image)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224]) / 255.
        return img

    def config_for_image_performance(self, ds):
        ds = ds.cache()
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=self.autotune)
        return ds

    def build_image_loader(self):
        train_ds = tf.data.Dataset.list_files(self.image_paths, shuffle=False)
        train_ds = train_ds.map(self.processing_image, num_parallel_calls=self.autotune)

        # Performance
        return self.config_for_image_performance(train_ds)


if __name__ == '__main__':
    meta_dir = "dataset/Flicker8k_Dataset/"
    meta_file = "dataset/Flickr8k.token.txt"

    loader = DatasetLoader(meta_dir, meta_file, batch_size=1)
    token = loader.load_tokenizer()
    sequences_padded = loader.build_capture_loader()
    sentence = next(iter(sequences_padded))
    # print(sentence)

    train_ds = loader.build_image_loader()
    image_batch = next(iter(train_ds))
    # print(image_batch)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        label = sentence[i]
        plt.title(token.sequences_to_texts(sentence.numpy())[i])
        plt.axis("off")
        plt.show()
