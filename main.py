import os

from PIL import Image

from loader import DatasetLoader

meta_dir = "dataset/Flicker8k_Dataset/"
meta_file = "dataset/Flickr8k.token.txt"

loader = DatasetLoader(meta_dir, meta_file)
loader.loader()

for img_path in loader.image_paths:
    im = Image.open(img_path)
    save = "dataset/jpegs/" + os.path.basename(img_path).split(".")[0] + ".jpeg"
    if not os.path.exists(save):
        im.save(save)
