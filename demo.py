class loader:
    def __init__(self, list_data=[], batch_size=32):
        self.bs = batch_size
        self.list_data = list_data
        self.index = 0

    def __next__(self):
        train_ds = self.list_data[self.index: self.index * self.bs]
        self.index += bs
        return train_ds


def load_img(img):
    return


def rotate_img(img):
    return


if __name__ == '__main__':
    data = loader()
    list_data = []
    for bs in range(len(list_data)):
        data_train = data.__next__()
