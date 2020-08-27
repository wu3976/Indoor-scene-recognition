'''from PIL import Image


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


path = "./data/train/airport_inside/airport_inside_0001.jpg"

img = Image.open(path)
new_img = crop(img, img.size)
new_img.save("./test.jpg")

img2 = Image.open("./test.jpg")
new_img_2 = crop(img2)
new_img_2.show()
save_img = new_img_2.resize((500, 500))
save_img.save("./final_img.jpg")'''

import matplotlib.pyplot as plt
import torch
import math
from model import WideResnet
'''x = [1 * i for i in range(50)]
y = []
z = []
for i in range(50):
    y.append(pow(i, 2))
    z.append(math.sin((1/2) * i))

print(y)

print(x)
plt.plot(x, y, label="validation loss")
plt.xlabel("epoches")
plt.ylabel("loss")
plt.show()

plt.plot(x, z, label="validation accuracy")
plt.xlabel("epoches")
plt.ylabel("accuracy")
plt.show()'''

model = WideResnet()

torch.save(model.state_dict(), "./test.pt")

model2 = WideResnet()
model2.load_state_dict(torch.load("./test.pt"))

print("loaded!")
'''avg_validation_losses = [12.0, 23.0, 34.0]
avg_validation_accuracy = [0.1, 0.2, 0.3]

file_loss = open("./loss.txt", "w+")
file_loss.writelines([str(ele) + "\n" for ele in avg_validation_losses])
file_accr = open("./accuracy.txt", "w+")
file_accr.writelines([str(ele) + "\n" for ele in avg_validation_accuracy])'''