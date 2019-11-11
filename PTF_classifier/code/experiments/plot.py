import os
import numpy
import generate
import matplotlib.pyplot as plt
from util import ensure_dir

examples_per_class = 50

X_2d, X_features, y, indices = generate.get_data("all")
print("Total number of instances: " + str(len(y)))
distribution = numpy.bincount(y)
distribution = distribution[1:]
plt.bar(range(1,18), distribution, align='center')
ensure_dir("plots/distribution.png")
plt.savefig("plots/distribution.png", bbox_inches='tight')
plt.close()


# classes start from 0 (not from 1 as in the data)
for cl in range(0,17):

    real_class = cl + 1
    print("Processing class %i ..." % real_class)

    if not os.path.exists(os.path.join("plots", str(real_class))):
        os.makedirs(os.path.join("plots", str(real_class)))

    for i in range(examples_per_class):

        try:
            ofname = os.path.join("plots", str(real_class), str(i) + ".png")
            data = X_2d[y==real_class][i]
            plt.imshow(data)
            plt.savefig(ofname, bbox_inches='tight')
            plt.close()
        except:
            pass    

    # generate mean image (based on ALL instances per class)       
    ofname = os.path.join("plots", str(real_class) + "_mean.png")
    data = X_2d[y==real_class].mean(axis=0)
    plt.imshow(data)
    plt.savefig(ofname, bbox_inches='tight')
    plt.close()


