# Reproducibility
Experimenting with different techniques for reproducibility in deep learning.

The project deals with the following:

* pix2pi**XAI**: Generating class-specific visualizations from input-specific visualizations like grad-CAM, Saliency Maps and SHAP using pix2pix Generative Adversarial Network
* bra**XAI**: Interpreting [braai](https://github.com/dmitryduev/braai) performing Real-Bogus classification for the Zwicky Transient Facility (ZTF) using Deep learning. 
* Classification of Periodic Variables present in Catalina Real-Time Transient Survey(CRTS) using Interpretable Convolutional Neural Networks
* Transfer Learning: Predicting unclassified variables observed and recorded by Zwicky Transient Facility (ZTF) and Palomar Transient Facility (PTF) using the Convolutional Neural Network trained using CRTS data
* workflow: Creating a workflow for periodic variable prediction from the light curve data directly using dmdt generation and trained CNN

## Getting Started

### Prerequisites


* Python 3
* Tensorflow >= 1.0
* Keras > 2.0
* [keras-vis](https://github.com/raghakot/keras-vis)
* [shap](https://github.com/slundberg/shap)
* [tcav](https://github.com/tensorflow/tcav)
* [scikit-image](https://scikit-image.org/)
* [AIX360](https://github.com/IBM/AIX360)
* [pix2pix](https://github.com/affinelayer/pix2pix-tensorflow)
* [nolearn](https://pythonhosted.org/nolearn/)
* [lasagne](https://github.com/Lasagne/Lasagne)
* [theano](http://deeplearning.net/software/theano/)

### Installation

```
pip install keras-vis
pip install shap
pip install tcav
pip install nolearn
pip install Lasagne==0.1
git clone git://github.com/Theano/Theano.git
git clone https://github.com/IBM/AIX360
git clone https://github.com/affinelayer/pix2pix-tensorflow.git
https://github.com/amiratag/ACE.git
```

## Classification of Light Curves

![alt text](https://github.com/AshishMahabal/Reproducibility/blob/master/doc/lightcurves.png "Light curves")

A light curve is a time series dataset of magnitude, the negative logarithm of flux measurement(as smaller magnitude implies brighter objects). The measurements available in these light curve datasets are:

* Right Ascension(RA) and Declination(Dec) which provide the position of the object on the sky
* Time reference(epoch) as Julian Date
* Magnitude
* An error estimate on the magnitude

Most of the data collected from astronomical surveys are sparse, far from continuous, mostly irregular and heteroscedastic.

Astronomical objects exhibit variation in brightness due to some intrinsic physical process like explosion and or merger of matter inside or due to some extrinsic process like eclipse or rotation. These astronomical objects are termed as **Variables**. Variables with brightness varying by several standard deviations for a very short period of time are called **Transients**.

### Light curve data processing into *dmdt*

![alt text](https://github.com/AshishMahabal/Reproducibility/blob/master/doc/lightcurve2dmdt.png "Light curve to dmdt")
![alt text](https://github.com/AshishMahabal/Reproducibility/blob/master/doc/dmdt.png "dmdt")

A light curve is transformed into a 2D mapping based on changes in magnitude *dm* and time differences *dt*, so that they can be used as an input to Convolutional Neural Network. Note that to give each bin in a dmdt an equal footing, the dmdt bins are of same size instead of the bin spacing depending on the actual magnitude of *dm* and *dt*. To know more about *dmdts*, refer [Deep-Learnt Classification of Light Curves](https://arxiv.org/abs/1709.06257)

### Light curve data preparation and *dmdt* generation

Follow [DATA.md](https://github.com/AshishMahabal/Reproducibility/blob/master/Periodic%20Variable%20Classification/data/v0/indiv_lc/DATA.md) on how the light curve data as **.csv** should be placed in **Periodic Variable Classification/data** folder.

Once the light curve data is placed in the appropriate folder configuration, run [transform.py](https://github.com/AshishMahabal/Reproducibility/blob/master/Periodic%20Variable%20Classification/data/transform.py) to generate dmdts and appropriate labels in **Periodic Variable Classification/data/all** folder.

### Training a CNN for classification of *dmdts*

![alt text](https://github.com/AshishMahabal/Reproducibility/blob/master/doc/periodic_classes.png "Periodic Classes")

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Iu5t6AeDhaqgecNV56dqq5pKaf6Sb_Lk)

## XAI

### Input-specific interpretations

#### [Gradient-weighted Class Activation Mapping](http://gradcam.cloudcv.org/)

*grad-CAM is another way of visualizing attention over input which uses penultimate (pre Dense layer) Conv layer output. The intuition is to use the nearest Conv layer to utilize spatial information that gets completely lost in Dense layers.* - [Class Activation Maps](https://raghakot.github.io/keras-vis/visualizations/class_activation_maps/)

![alt text](https://github.com/AshishMahabal/Reproducibility/blob/master/CNN%20interpretation/keras-vis/grad_CAM/EW/1005.png "gradCAM of 1005th test dmdt")

#### [Saliency Maps](https://arxiv.org/pdf/1312.6034v2.pdf)

*Saliency Maps are generated by computing the gradient of output category with respect to input image which would tell us how output category value changes with respect to a small change in input image pixels. All the positive values in the gradients tell us that a small change to that pixel will increase the output value. Hence, visualizing these gradients, which are the same shape as the image should provide some intuition of attention.* - [Saliency Maps](https://raghakot.github.io/keras-vis/visualizations/saliency/)

![alt text](https://github.com/AshishMahabal/Reproducibility/blob/master/CNN%20interpretation/keras-vis/saliency/EW/1005.png "Saliency Map of 1005th test dmdt")

#### Blanking experiments

The basic approach of this method is to blank out each pixel of the dmdt image and then predict the changes in prediction probabilities of that dmdt image belonging to the particular class which it belonged to before blanking it out.

![alt text](https://github.com/AshishMahabal/Reproducibility/blob/master/doc/4_median_heatmap.png "Blanking exp. visualization")

#### [SHAP](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)

*Deep SHAP is a high-speed approximation algorithm for SHAP values in deep learning models that builds on a connection with [DeepLIFT](https://arxiv.org/abs/1704.02685).* - [shap](https://github.com/slundberg/shap)

![alt text](https://github.com/AshishMahabal/Reproducibility/blob/master/doc/SHAP_EW_44.png "SHAP visualization of 44th test dmdt")

Note that in the above SHAP visualization of 44th test dmdt, hotter pixels increase the model's output while cooler pixels decrease the output.

### Class-specific interpretations

The visualizations that we had generated up until now i.e. grad-CAM, Saliency Attention Maps, Blanking Exp. and SHAP are all input-specific visualizations; hence for each test data, there would be a corresponding above four visualization plots. However, for interpretability, it would be helpful if these plots are generated class-specific instead of input-specific. 

#### pix2piXAI

![alt text](https://github.com/AshishMahabal/Reproducibility/blob/master/XAI/pix2pixai.png "pix2piXAI Methodology")

To generate a class-specific visualization from several input-specific visualizations, we have formulated a technique in which we pass the test dmdts along with their corresponding visualization through a pix2pix GAN for training. The motivation behind the same is that after training, the generator of pix2pix GAN will learn the most relevant features from the visualization plots; after which the test data with prediction probability greater than 0.95 is passed as test data to the trained pix2pix GAN which will generate corresponding visualization plots. A similarity metric is then used to find the visualization plot most learnt by the pix2pix GAN and hence we get class-specific visualizations.

### To see class-specific interpretations generated by pix2piXAI, [click here](https://github.com/AshishMahabal/Reproducibility/tree/master/XAI/pix2piXAI%20class-specific%20visualization%20plots) 

#### Lighting Experiments

![alt text](https://github.com/AshishMahabal/Reproducibility/blob/master/XAI/lighten/EA.png "Lighting Exp. for EA")

The basic approach of this method is to create an instance/image such that just one pixel is lightened (i.e. pixel value = 255). Thereafter this image is passed through already trained CNN which outputs the prediction probabilities. These prediction probabilities map into each of the 2D matrix pertaining to each of the class whose shape is similar to the image. Hence repeating the above process for each pixel of the original image, we get a probability mapping from one lightened pixel into a number of 2D matrices pertaining to each of the classes. This analysis allowed us to visualize which pixel is important for classification when lightened more (i.e. more number of objects lie in that pixel).

### To see class-specific interpretations generated by Lighting Experiments, [click here](https://github.com/AshishMahabal/Reproducibility/tree/master/XAI/lighten)

#### Activation Maximization

![alt text](https://github.com/AshishMahabal/Reproducibility/blob/master/XAI/activation%20maximization/preds/RRab.png "Activation Maximization for pred layer's filter corresponding to class RRab")

*In a CNN, each Conv layer has several learned template matching filters that maximize their output when a similar template pattern is found in the input image. Activation Maximization generates an input image that maximizes the filter output activations. This allows us to understand what sort of input patterns activate a particular filter.* - [Activation Maximization](https://raghakot.github.io/keras-vis/visualizations/activation_maximization/)

### To see class-specific interpretations generated by Activation Maximization, [click here](https://github.com/AshishMahabal/Reproducibility/tree/master/XAI/activation%20maximization/preds)

## braXAI

braXAI concerns with interpreting [braai](https://github.com/dmitryduev/braai) which performs Real-Bogus classification for the Zwicky Transient Facility (ZTF) using Deep learning. The only difference between our and braai's VGG model is that in our VGG model **fc_out** (output) layer contains two neurons instead of one in braai's VGG model. These two neurons in the outer layer correspond to the two classes which allows for visualizations pertaining to both the classes instead of visualizations of just one of the class in braai's VGG model case; hence two neurons eases interpretation.

See this [Jupyter Notebook](https://github.com/AshishMahabal/Reproducibility/blob/master/braxai/braxai_train.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iO27WMgNqZaROeD2hwUOwLDKvjyHIhgp)

### Class-specific interpretations

#### pix2piXAI

Class-specific interpretations were generated for braXAI using pix2piXAI in the same manner as earlier(for Periodic Variable classification). However based on the application, real and bogus classes were further divided into subcategories for more refined interpretation

* Real Class:
    - real_central: a central roughly round peak
    - real_modulo_noise: a mostly blank image other than the centre (modulo noise)
    - real_mixture: images with both a central peak and modulo noise
* Bogus Class:
    - bogus_blanked: removing artefacts by blanking
    - bogus_non_blanked: a bogus image with no blanked portion in the image

Class-specific interpretations for braXAI generated by pix2piXAI corresponds to gradCAM, Saliency Maps, Blanking Exp. and SHAP(of DIFF/SUB) visualizations for each of the above subcategories.

![alt text](https://github.com/AshishMahabal/Reproducibility/blob/master/braxai/class_specific_visualizations/bogus_blanking_blanked.png "Blanking Exp. candidate visualization for bogus class' blanked subcategory")

### To see all class-specific interpretations generated by pix2piXAI for braXAI, [click here](https://github.com/AshishMahabal/Reproducibility/tree/master/braxai/class_specific_visualizations)

#### Activation Maximization

Similar to Periodic Variable classification, Activation Maximization maps are generated for real and bogus classes based on **fc_out** (output) layer. Below are the corresponding two interpretations.

![alt text](https://github.com/AshishMahabal/Reproducibility/blob/master/braxai/activation_maximization/fc_out/real_filter.png "Real Class filter in fc_out layer")

<!---![alt text](https://github.com/AshishMahabal/dmdt_trans/blob/master/braxai/activation_maximization/fc_out/bogus_filter.png "Bogus Class filter in fc_out layer")-->

### Misclassifications

To get more insight, we have generated visualizations for misclassifications as well. In total there were 45 misclassifications out of 1156 test images. Each misclassification plot contains SCI, REF, DIFF, grad_CAM, saliency, blanking, SHAP of SCI, SHAP of REF and SHAP of DIFF/SUB visualizations. Note that every visualization is with respect to the predicted class(mentioned in the plot's title).

![alt text](https://github.com/AshishMahabal/Reproducibility/blob/master/braxai/misclassifications/1015.png "Misclassification Plot for 1015th test image")

### To see all 45 misclassification plots, [click here](https://github.com/AshishMahabal/Reproducibility/tree/master/braxai/misclassifications)

<!---
## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

-->

<!---
## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
-->

## Authors

* Ashish Mahabal - [Website](http://www.astro.caltech.edu/~aam/)
* Meet Gandhi
  - If you encounter any problems/bugs/issues please contact me on Github or by emailing me at gandhi.meet@btech2015.iitgn.ac.in for any bug reports/questions/suggestions. I prefer questions and bug reports on Github as that provides visibility to others who might be encountering same issues or who have the same questions.

## Acknowledgments

* Raghavendra    Kotikalapudi    and    contributors.keras-vis.https://github.com/raghakot/keras-vis, 2017.

* Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks.CVPR, 2017.

* Scott M Lundberg and Su-In Lee. A unified approach to interpreting model predictions. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems 30, pages 4765–4774. Curran Associates, Inc., 2017.

* Mahabal, Ashish, et al. "Deep-learnt classification of light curves." 2017 IEEE Symposium Series on Computational Intelligence (SSCI). IEEE, 2017.

* Dmitry A Duev, Ashish Mahabal, Frank J Masci, Matthew J Graham, Ben Rusholme,  Richard  Walters,  Ishani  Karmarkar,  Sara  Frederick,  Mansi  M Kasliwal,  Umaa  Rebbapragada,  et  al.    Real-bogus  classification  for  the zwicky transient facility using deep learning. Monthly Notices of the Royal Astronomical Society, 489(3):3582–3590, 2019.
