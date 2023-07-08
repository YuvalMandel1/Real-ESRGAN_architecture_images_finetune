# Real-ESRGAN architecture images finetune
In this project we aim to fine-tune the Real-ESRGAN model to handle real-life architecture images in a better way.
## Introduction
Single image super-resolution is an active research topic, which aims at reconstructing a high-resolution image from its low-resolution counterpart. One of the current best models in the field is Real-ESRGAN (Real Enhanced Super Resolution GAN), which uses a more advanced degradation process than its predecessors. One of the limitation with Real-ESRGAN and other models is the loss of details and textures important for real-life images, Because of the training on more simple, texture-less images like animation. In this project we aim to fine-tune the Real-ESRGAN model to handle real-life architecture images in a better way.
## Previous work
The Real-ESRGAN model is the product of several successful research attempts, starting with SRGAN as the base architecture, which presented the usage of GAN architecture with a Generator model for the image upscale, and a Discriminator model which learns to classify real high-resolution images against artificial images.

![alt text](https://github.com/YuvalMandel1/Real-ESRGAN_architecture_images_finetune/images/SRGAN_model.png)

The next step in the model evolution was the introduction of ESRGAN. The new model updated the Generator model so it is composed the following way:
1.	A basic component is the RDB (Residual Dense Block). It has an input size of 64 channels, which is composed of 5 convolution layers, (3x3 kernel, stride 1 and padding 1) and a Leaky-ReLU layer. Each layer receives the all of the inputs of the layers that were previous to it (Skip connection) and the output of the layer before it, and has an output of 32 channels, except the final layer which has an output of 64 channels.
2.	The component that wraps the RDB is the RRDB (Residual-in-Residual Dense Block, also known as Basic Block). It holds 3 RDB’s with skip connections.
3.  The final architecture is composed first with:
    1)	An input convolution layer of 3 channels to 64 channels, with 3x3 kernel, stride 1 and padding 1.
    2) 23 RRDB units connected sequentially.
    3)	A transfer convolution layer which receives the output of the last RRDB unit, and transfer it to the up-sampling layers. This layer is a 64 to 64 channels, with 3x3 kernel, stride 1 and padding 1.
    4)	2 up-sampling convolution layers, each is 64 to 64 channels, 3x3 kernel, stride 1 and padding 1.
    5)	High resolution layer. This layer is a 64 to 64 channels, with 3x3 kernel, stride 1 and padding 1.
    6)	An output convolution layer. This layer is a 64 to 3 channels, with 3x3 kernel, stride 1 and kernel 1.

![alt text](https://github.com/YuvalMandel1/Real-ESRGAN_architecture_images_finetune/images/ESRGAN_generator.png)

The ESRGAN also introduced a new Discriminator model. Rather than estimating the probability that an image is either natural or synthetic, this discriminator attempts to predict which is the natural image, between a natural image or a fake upscale photo the generator created.

![alt text](https://github.com/YuvalMandel1/Real-ESRGAN_architecture_images_finetune/images/ESRGAN_discriminator.png)

