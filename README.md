# DETR Model from Scratch
In an effort to better understand CNNs and Transformers, I am building the DETR model from scratch. This model has 141M parameters and can be trained on Google Colab.

## Model Overview

1. CNN: The model starts with a CNN that outputs a series of channels representing features of size (H, W).
2. Positional Encodings: These features are combined with positional encodings and fed into the encoder.
3. Encoder: The encoder processes the features and passes them to the decoder.
4. Decoder: The decoder processes the input and passes it through a classifier layer to produce raw outputs.
5. Loss Function: The final outputs are processed with a softmax function, and a minimum loss algorithm is applied to match predictions with labels. The loss function used is the negative log likelihood (Cross-entropy).

## Python Scripts
There are three main Python scripts, each representing different levels of abstraction when training neural networks:

1. detr_scratch.py:

Implements the DETR model from scratch using minimal abstractions from the PyTorch API.
All matrix multiplications, linear/non-linear transformations, and CNN kernels are implemented using math and tensors.
This script is an exercise to build intuition about the underlying operations, which is crucial for understanding and debugging neural networks.

2. detr_pytorch.py:
Replaces many manual operations with higher-level abstractions from PyTorch's API.
This level of abstraction is typical for building custom models in practice.

3. detr_finetuned.py:
Removes the layer-defining code and fine-tunes an already trained DETR model.

## Results
The results compare the training performance of the three approaches on the same dataset. This comparison is not definitive but provides insights into the differences between fine-tuning and implementing a model from scratch.

[Results]

Feel free to explore the scripts and results to gain a deeper understanding of the DETR model and its implementation from scratch.