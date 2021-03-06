


# Adversarial sample generation for the detection of  adversarial attacks

In this repository You can find our code which implements a simple example for adversarial attack generation on the MNIST dataset containing two classes only. This code is supplementary material for our submitted paper:

*Detection of sticker based adversarial attacks*
Csanád Egervári, András Horváth

Submitted to:
International Conference on Digital Image Processing

### Prerequisites-Installing

To run our code You need to install [Python](https://www.python.org/)  (*v3.5*) and  [Tensorflow](https://www.tensorflow.org/) (v1.3.0) and that is all.

### Running our code
Our data set for this example (a subset of the MNIST dataset can be found in the Folder Data as numpy arrays).
 Our training scripts were implemented as a single file. training the original network can be done by executing [train_network.py](https://github.com/horan85/adversarialsticker/blob/master/train_network.py).
Low intensity adversarial noise, covering the whole image can be trained by [low_intensity_noise.py](https://github.com/horan85/adversarialsticker/blob/master/low_intensity_noise.py) and sticker based adversarial samples can be generated by [sticker_noise.py](https://github.com/horan85/adversarialsticker/blob/master/sticker_noise.py).
We have to note that these are examples for adversarial sample generation on a simple dataset, the position and size of the stickers can be optimized by genetic algorithm.

## Example
In our submitted paper we have shown that:

- low intensity adversarial attacks, covering the whole image are not practical. Low intensity noise is not robust enough and gets distorted by perspective distortion or even by the optical system
- sticker based attack, concentrated on the small size of the image can be implemented in real life, but can easily be detected by checking the response of certain features at different regions in the network (because a large response will come from a small region) or they can be detected by visualizing the network response and on can easily detect the classification is not consistent and altered by two regions.

![alt text](https://github.com/horan85/adversarialsticker/raw/master/fig/sticker_sample.png)
![alt text](https://github.com/horan85/adversarialsticker/raw/master/fig/heatmap.png)

## Authors
Csanád Egervári
András Horváth 
