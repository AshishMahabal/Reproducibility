import numpy as np
import matplotlib.pyplot as plt
import os

x_train = np.load("x_train.npy")
x_test = np.load("x_test.npy")

y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

clss = {0:"bogus", 1:"real"}

os.mkdir("ace/")
os.mkdir("ace/SCI/")
os.mkdir("ace/REF/")
os.mkdir("ace/SUB/")
os.mkdir("ace/SCI/real/")
os.mkdir("ace/SCI/bogus/")
os.mkdir("ace/REF/real/")
os.mkdir("ace/REF/bogus/")
os.mkdir("ace/SUB/real/")
os.mkdir("ace/SUB/bogus/")

x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))

np.save("ace/x.npy", x)
np.save("ace/y.npy", y)

for i in range(x.shape[0]):
    plt.figure()
    plt.imshow(x[i][:, :, 0], origin='upper', cmap=plt.cm.bone)
    plt.axis("off")
    plt.savefig("ace/SCI/"+clss[y[i]]+"/"+str(i)+".png",
              bbox_inches='tight', transparent="True", pad_inches=0)
    plt.close()

    plt.figure()
    plt.imshow(x[i][:, :, 1], origin='upper', cmap=plt.cm.bone)
    plt.axis("off")
    plt.savefig("ace/REF/"+clss[y[i]]+"/"+str(i)+".png",
              bbox_inches='tight', transparent="True", pad_inches=0)
    plt.close()

    plt.figure()
    plt.imshow(x[i][:, :, 2], origin='upper', cmap=plt.cm.bone)
    plt.axis("off")
    plt.savefig("ace/SUB/"+clss[y[i]]+"/"+str(i)+".png",
              bbox_inches='tight', transparent="True", pad_inches=0)
    plt.close()

x_random_indices = np.random.choice(x.shape[0], size = 500, replace = False)
np.save("ace/x_random_indices.npy", x_random_indices)

os.mkdir("ace/SCI/random/")
os.mkdir("ace/REF/random/")
os.mkdir("ace/SUB/random/")

for i in x_random_indices:
    
    plt.figure()
    plt.imshow(x[i][:, :, 0], origin='upper', cmap=plt.cm.bone)
    plt.axis("off")
    plt.savefig("ace/SCI/random/"+str(i)+".png",
              bbox_inches='tight', transparent="True", pad_inches=0)
    plt.close()

    plt.figure()
    plt.imshow(x[i][:, :, 1], origin='upper', cmap=plt.cm.bone)
    plt.axis("off")
    plt.savefig("ace/REF/random/"+str(i)+".png",
              bbox_inches='tight', transparent="True", pad_inches=0)
    plt.close()

    plt.figure()
    plt.imshow(x[i][:, :, 2], origin='upper', cmap=plt.cm.bone)
    plt.axis("off")
    plt.savefig("ace/SUB/random/"+str(i)+".png",
              bbox_inches='tight', transparent="True", pad_inches=0)
    plt.close()
    
