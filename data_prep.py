from torchvision.datasets.utils import download_url
import tarfile, os
from PIL import Image


DIR_NAME = "train"
def download_and_extract(url, dir):
    download_url(url, root=".")
    with tarfile.open("indoorCVPR_09.tar") as tar:
        tar.extractall(dir)


def create_label_file(filepath, labels):
    file = open(filepath + "/labels.txt", mode="w+")
    lines = [label + "\n" for label in labels]
    file.writelines(lines)


def crop(img):
    width, height = img.size
    # crop test
    new_img = None
    if width > height:
        # top and bottom retains
        left = (width - height) / 2
        right = left + height
        new_img = img.crop((left, 0, right, height))
    else:
        # left and right retains
        top = (height - width) / 2
        bottom = top + width
        new_img = img.crop((0, top, width, bottom))
    return new_img


def process_image(img_path, size):
    img = Image.open(img_path)
    img = crop(img)
    img = img.resize(size)
    return img


def process_images(dir, size):
    imagename_list = os.listdir(dir)
    print(dir)
    for img_name in imagename_list:
        img_path = dir + "/" + img_name
        new_img = process_image(img_path, size)
        new_img = new_img.convert("RGB")
        new_img.save(img_path)


if __name__ == "__main__":
    LABEL_FILE_DIR = "./data"
    IMAGE_DIR = "./data/" + DIR_NAME
    # download_and_extract("http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar", "./data")
    labels = os.listdir(IMAGE_DIR)
    print(labels)

    for dir in labels:
        process_images(IMAGE_DIR + "/" + dir, (256, 256))

    create_label_file(LABEL_FILE_DIR, labels)
