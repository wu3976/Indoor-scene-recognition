'''file = open("./temp.txt", "r")
vloss = open("loss2.txt", "w+")
vacc = open("accuracy2.txt", "w+")
while True:
    line = file.readline()
    print("Line: " + line)
    if line == "":
        break
    elif line.find("validation loss") != -1:
        data = line[line.index("(") + 1: line.index(",")]
        vloss.write(data)
        vloss.write("\n")

    elif line.find("validation accuracy") != -1:
        data = line[line.index("(") + 1: line.index(")")]
        vacc.write(data)
        vacc.write("\n")'''

import matplotlib.pyplot as plt

acc_file = open("accuracy2.txt", "r")
loss_file = open("loss2.txt", "r")

acc = acc_file.readlines()
acc = [float(ele[0:len(acc)]) for ele in acc]
loss = loss_file.readlines()
loss = [float(ele[0:len(loss)]) for ele in loss]

print(acc)
print(loss)
x = range(50)

plt.plot(x, acc)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("Accuracy vs. time")
plt.show()

plt.plot(x, loss)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Loss vs. time")
plt.show()