In an effort to better understand CNNs and Transformers I am building the DETR model from scratch as it's only a small 141M parameters and can be trained in a google colab. 


It has a initially CNN that outputs a series of channels representing features of size (H, W). These are then combined with positional encodings to feed into a encoder. 

The ecoder does some stuff and pushes it to the decoder. After the decoder gets done decoding, it passes it through a classifier layer and then to a raw output. Softmax is applied, 
and the minimum loss algo is applied to match predictions with labels to define a loss function. The loss function we will use is the negative log likelihood (Cross-entropy).