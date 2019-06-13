# Modified layers used in the model

Things I have tested:

![x] LSTM (computation is slower than GRU)
![x] 3d Conv (MSE is high)
![x] Nalu
![x] Combining Nalu with conv
![x] Combining Nalu with Coordconv 
![x] Combining LSTM with conv


Eventually,

I combined Coordconv with NALU to reduce the limitation of convolution neural network.
Also combined convolution with GRU so that the model is able to learn time series data.

### NALU

[Research papers on NALU](https://arxiv.org/pdf/1808.00508.pdf)


### Coordconv

[Research papers on Coordconv](https://arxiv.org/pdf/1807.03247.pdf)

