# DETR Model from Scratch: Table of Contents

1. [Introdiction](#introduction)
2. [Model Overview](#model-overview)
3. [Python Script Overview](#python-script)
4. [Tuning Hyperparameters](#tuning-hyper-parameters)
5. [Results](#results)
6. [Improvements](#improvements)

## Introduction
In an effort to better understand CNNs and Transformers, I am building the DETR model from 'scratch' (Note: The transformer is really the main thing built from scratch. The CNN, Linear and MLP layers were initialized using the PyTorch API). 


1. Get a count of the number of parameters. 

This model has 141M parameters and can be trained on Google Colab. The code in the .py files will have far, far too many comments for any repo - and I like a lot of comments lol - but I made them mainly as my own note taking and references. 

If I had more money I would get some GPU compute and try to reproduce the DETR paper; however, this is a very expensive, data hungry architecture as it uses transformers. Nonetheless, we were still able to train from scratch and get reasonable results. With more compute, I'd be able to set up many more tests to tweak the hyper parameters. 

[GPU Poor Meme]

## Model Overview

1. CNN: The model starts with a CNN that outputs a series of channels representing features of size (H, W).
2. Positional Encodings: These features are combined with positional encodings and fed into the encoder.
3. Encoder: The encoder processes the features and passes them to the decoder.
4. Decoder: The decoder processes the input and passes it through a classifier layer to produce raw outputs.
5. Loss Function: The final outputs are processed with a softmax function, and a minimum loss algorithm is applied to match predictions with labels. The loss function used is the negative log likelihood (Cross-entropy).

## Python Script:
Implements the DETR model from scratch using minimal abstractions from the PyTorch API.
All matrix multiplications, linear/non-linear transformations, and CNN kernels are implemented using math and tensors.
This script is an exercise to build intuition about the underlying operations, which is crucial for understanding and debugging neural networks.

## Tuning Hyper parameters

<p align="center">
  <img src="charts/backbone_v1.png" alt="Diagram" width="350"/>
  <img src="charts/fnn_bbox_mlp_layer1_v1.png" alt="Diagram" width="350"/>
</p>

As you can see the from these two examples - see the v1 folder contains the weight distribution across more layers of the network - the distribution of weights at initialization are small and uniform landing between -0.02 and 0.02. The distribution is good for initial weights as they aren't too large that the first few training iterations are spent condensing the weights to an 'appropriate scale' and don't have a distribution far outside the appropriate mean of 0 and standard deviation of 1. 

Initializing very small, uniform weights can be extremely effective for training reliability in shallow neural networks; however, as the depth of the network grows it's extremely hard to fight against exploding or vanishing gradients. One trick is to add residual layers that add to the layer outputs. The reason this works so well is the gradient computed from the loss distributes evenly through addition (i.e the same gradient goes through the residual connection and deep learning layers). Therefore, the gradient doesn't vanish or explode as it's not reliant on any one neural network layer to properly process and distribute the gradient (i.e not one layer acts as the gate keeper to the initial gradient). 

[Gradients over time]

So by initializaing the layer weights near 0, the gradient slowly 'turns these layers on' and adjusts the parameters of the model from the ground up as the gradient from the loss 'flows' straight to the first layer (i.e object queries and backbone). 

[Weights sum over time]


## Results
The results compare the training performance of the three approaches on the same dataset. This comparison is not definitive but provides insights into the differences between fine-tuning and implementing a model from scratch.

[Results]

Feel free to explore the scripts and results to gain a deeper understanding of the DETR model and its implementation from scratch.

## Improvements 
1. More GPUs 
2. Better initialization strategy. Based on what I have read, He initialization seems to be a good choice from deep networks utilizing ReLu as the non-linear function. 
3. Import the resnet-50 backbone. In the ["End-to-End Object Detection with Transformers"](https://arxiv.org/abs/2005.12872) research paper, they import the resnet-50 backbone instead of initializing backbone layers (i.e CNNs, Positional Encodings, etc) from scratch. 
4. This model takes a lot of compute, so as 
