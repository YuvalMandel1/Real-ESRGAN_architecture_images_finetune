# Real-ESRGAN Architecture Images Finetune
<p align="center">
    046211- Deep Learning Course, ECE Faculty, Technion
    <br>
    <br>
    :tiger: Tomer Richter Barouch: <a href="https://www.linkedin.com/in/tomerrb/">LinkdIn</a> , <a href="https://github.com/tomerrb">GitHub</a>
    <br>
    :wolf: Yuval Mandel: <a href="https://www.linkedin.com/in/yuval-mandel/">LinkdIn</a> , <a href="https://github.com/YuvalMandel1">GitHub</a>
</p>

In this project we aim to fine-tune the Real-ESRGAN model to handle real-life architecture images in a better way.
## :brain: Introduction
Single image super-resolution is an active research topic, which aims at reconstructing a high-resolution image from its low-resolution counterpart. One of the current best models in the field is Real-ESRGAN (Real Enhanced Super Resolution GAN), which uses a more advanced degradation process than its predecessors. One of the limitation with Real-ESRGAN and other models is the loss of details and textures important for real-life images, Because of the training on more simple, texture-less images like animation. In this project we aim to fine-tune the Real-ESRGAN model to handle real-life architecture images in a better way.
## :woman_teacher: Previous work
The Real-ESRGAN model is the product of several successful research attempts, starting with SRGAN as the base architecture, which presented the usage of GAN architecture with a Generator model for the image upscale, and a Discriminator model which learns to classify real high-resolution images against artificial images.

<p align="center">
    <img src="https://github.com/YuvalMandel1/Real-ESRGAN_architecture_images_finetune/blob/main/images/SRGAN_model.png">
</p>



The next step in the model evolution was the introduction of <a href="https://github.com/xinntao/ESRGAN/tree/master">ESRGAN</a>. The new model updated the Generator model so it is composed the following way:
1.	A basic component is the RDB (Residual Dense Block). It has an input size of 64 channels, which is composed of 5 convolution layers, (3x3 kernel, stride 1 and padding 1) and a Leaky-ReLU layer. Each layer receives the all of the inputs of the layers that were previous to it (Skip connection) and the output of the layer before it, and has an output of 32 channels, except the final layer which has an output of 64 channels.
2.	The component that wraps the RDB is the RRDB (Residual-in-Residual Dense Block, also known as Basic Block). It holds 3 RDB’s with skip connections.
3.  The final architecture is composed first with:
    1)	An input convolution layer of 3 channels to 64 channels, with 3x3 kernel, stride 1 and padding 1.
    2) 23 RRDB units connected sequentially.
    3)	A transfer convolution layer which receives the output of the last RRDB unit, and transfer it to the up-sampling layers. This layer is a 64 to 64 channels, with 3x3 kernel, stride 1 and padding 1.
    4)	2 up-sampling convolution layers, each is 64 to 64 channels, 3x3 kernel, stride 1 and padding 1.
    5)	High resolution layer. This layer is a 64 to 64 channels, with 3x3 kernel, stride 1 and padding 1.
    6)	An output convolution layer. This layer is a 64 to 3 channels, with 3x3 kernel, stride 1 and kernel 1.

<p align="center">
    <img src="https://github.com/YuvalMandel1/Real-ESRGAN_architecture_images_finetune/blob/main/images/ESRGAN%20Generator.png">
</p>

The ESRGAN also introduced a new Discriminator model. Rather than estimating the probability that an image is either natural or synthetic, this discriminator attempts to predict which is the natural image, between a natural image or a fake upscale photo the generator created.

<p align="center">
    <img src="https://github.com/YuvalMandel1/Real-ESRGAN_architecture_images_finetune/blob/main/images/ESRGAN_discriminator.png">
</p>

The discriminator neural network is composed from 10 convolution layers, the following way:
1.	3 input channels, 64 output channels, 3x3 kernel, stride 1, padding 1.
2.	64 input channels, 128 output channels, 4x4 kernel, stride 2, padding 1 and no bias.
3.	128 input channels, 256 output channels, 4x4 kernel, stride 2, padding 1 and no bias.
4.	256 input channels, 512 output channels, 4x4 kernel, stride 2, padding 1 and no bias.
5.	512 input channels, 256 output channels, 3x3 kernel, stride 1, padding 1 and no bias.
6.	256 input channels, 128 output channels, 3x3 kernel, stride 1, padding 1 and no bias.
7.	128 input channels, 64 output channels, 3x3 kernel, stride 1, padding 1 and no bias.
8.	64 input channels, 64 output channels, 3x3 kernel, stride 1, padding 1 and no bias.
9.	64 input channels, 64 output channels, 3x3 kernel, stride 1, padding 1 and no bias.
10.	64 input channels, 1 output channels, 3x3 kernel, stride 1, padding 1

The <a href="https://github.com/xinntao/Real-ESRGAN">Real-ESRGAN</a> model, based on ESRGAN, replaced the Degradation process applied to high-resolution images to receive low-resolution images used for training and testing the model. The new degradation process is conducted with 4 functions: Blur, Resize (Down-sampling), Noise and JPEG Compression. Each of those 4 steps is performed twice, and lastly a 2D-sinc filter is employed to synthesize common ringing and overshoot artifacts. This process emulates real-life degradation processes performed by cameras, image editors, transmission over the internet and more.

<p align="center">
    <img src="https://github.com/YuvalMandel1/Real-ESRGAN_architecture_images_finetune/blob/main/images/Real-ESRGAN%20Degredation%20process.png">
</p>

## :man_scientist: Method
### :cd: Dataset Managment
We used a <a href="https://www.kaggle.com/datasets/tompaulat/modernarchitecture">Modern Architecture dataset</a> we found on Kaggle which contains ~50,000 images. This dataset includes images of interior and exterior buildings and apartments. It is contaminated by architectural floor-plan images, but they are in a minor amount. Images from same building/scene are located in close proximity next to each other in sub-folders, with a shared substring in the image title. In a way, this makes the images not completely i.i.d, so we chose to use sub-folders (which we know don’t share the same scene), for splitting the data to train, test and validation sets.
### :man_with_probing_cane: Choosing Training Layer
Due to hardware limitations, we could not train the entire model, so we chose to train a single layer for the fine-tuning process. The first challenge we encountered was choosing which layer location to fine tune. We composed a training scheme where we took a subset of the training and validation images and performed 4500 training iterations (Batch size of 12 and learning rate of 1e-4, the model default). 
The experiments we conducted were on the 2 possible layers:
1.	The first one is the front-most layer we could train without experiencing memory issues during the training. It was located in RRDB 2, RDB 3, layer 5.
2.	The second one is the final layer of the model.
All the other layers were “frozen”.

Between both ways, we observed better quantitative results on the validation images. We used the PSNR metric to decide which layer to continue training. The average PSNR over the validation sub-set 24.6833 for the first model (First layer activated) and 24.6878 for the second (Last layer activated.
After that we chose to focus only on the last layer.
### :running_man: Training The Last Layer
We continued to train the model for an extra 4500 iterations using the same subset, this time using the whole training and validation data set. We then saw a lack of improvement in the validation set, so we lowered the learning rate to 1e-5 and used the full training and validation sets. With the new settings, we continued to train for 15,000 iterations.
## :cherries: Results
### :art: Qualitative Results
We can observe the difference between images from our vs the original Real-ESRGAN model:

<p align="center">
    <img src="https://github.com/YuvalMandel1/Real-ESRGAN_architecture_images_finetune/blob/main/images/results%20patches.png">
</p>

While some changes may be hard to see for the naked eye, we can spot some changes for each image:
1.	While the golden model erases any trace of the original image shadows, we can observe that the fine-tuned model has been able to partially recover them.
2.	While the prominent small cloud is visible in the original image, the Real-ESRGAN model makes it less visible, but our model manages to show it a little more.
3.	The “tree eyes” in the sides of the image almost disappear in the original model, they are more dominant in the new one.
4.	The fine-tuned model managed to generate a more detailed image with better lines and textures than the original models.
5.	The colours and textures of the fine-tuned produced image are visibly better and more similar to the original image than the one produced by the Real-ESRGAN model.

### :dart: Quantitative Results
In terms of quantitative metrics, we saw the following results over the test set:

<table>
  <tr>
    <th>Metric</th>
    <th>Original</th>
    <th colspan="2">Real-ESRGAN</th>
    <th colspan="2">Fine-Tuned Model</th>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td>Score</td>
    <td>Difference</td>
    <td>Score</td>
    <td>Difference</td>
  </tr>
  <tr>
    <td>PSNR</td>
    <td> - </td>
    <td> 25.728 </td>
    <td> - </td>
    <td> 25.356 </td>
    <td> - </td>
  </tr>
  <tr>
    <td>NIQE</td>
    <td> 4.109 </td>
    <td> 5.825 </td>
    <td> 1.716 </td>
    <td> 5.768 </td>
    <td> 1.659 </td>
  </tr>
</table>

We can see a significant improvement of 0.057 in the average Naturalness Image Quality Evaluator (NIQE) metric, which compares the naturalness of an image compared to other natural images (This metric is an ML product that was learned on different natural images).
However, the average PSNR has gotten worse by 0.372, but since realistic architecture images are noisier than extra smooth images, which Real-ESRGAN produces, this metric is probably not the best way to evaluate our results, because the noise isn’t necessarily unwanted.

## :man_student: Conclusions
1.  Validate POC of architecture images: In this project we have proven that fine-tuning the Real-ESRGAN model to a specific domain of architecture images can improve its performance and similarity to natural architecture images.
2.  In general, fine-tuning a super-resolution model to a specific domain of images that share similar features, might be very beneficial for unseen images of that domain.
3.  The last layers have better and greater influence on the image restoration compared to the front-most layers, and for other domain fine-tuning we recommend using the available hardware resources to train mostly them for better model performance.

## :desktop_computer: Running the code
In order to run the code in this git repo, one must first do these steps:
1.	Download the original files from the Real-ESRGAN repo - https://github.com/xinntao/Real-ESRGAN.
2.	After downloading the files to "your_dir" Add the files from the architecture fine-tune repo in the Real-ESRGAN base folder *in the exact folders they are in both reposetories* ie:
    *  *Real_ESRGAN_architecture_finetune.ipynb* in the *yourt_dir* folder.
    *  *finetune_realesrgan_x4plus_architecture.yml* in the your_dir/options folder.
    *  *train_finetune_architecture.py* in the your_dir/realesrgan folder.
3.	We suggest to upload the updated folder to google drive and work with google colab, but it is possible to work with other tools with some modifications.
4. open the *Real_ESRGAN_architecture_finetune.ipynb* file and from then you can run the code, needed explnations are in the notebook.

## :rocket: Future Work
1.	Continue training, possibly trying more hyper parameters: The model might improve with more training as seen in previous sections. It is very possible that different hyper-parameters like a smaller learning rate, the usage of different schedulers will improve the seen results.
2.	With more hardware resources, training on more layers can be beneficial and improve the performance.
3.	Fine-tuning on sub domains of the architecture images, for example, indoor/outdoor, different architecture styles and more, can improve performance for those sub-domains.
4.	Fine-tuning a smaller model (like Real-ESRNet) can be faster with less hardware resources.
5.	Assimilate in a UI: Finally, after training and assembling the best model, an assimilation in a UI/App is bound to happen. This will be the finished product which could be applied to several tasks in the domains of photography and image editing, architecture and advertising.

## :eyes: See Also
:arrow_forward: <a href="https://www.youtube.com/watch?v=KZQvO0Va2wE&t=1s">Project Presention</a> (HEB)

## :raised_hands: References
:arrow_right: C. Ledig et al., “Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network,” Sep. 2016. [<a href="https://arxiv.org/abs/1609.04802v5">Arxiv</a>]

:arrow_right: X. Wang et al., “ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks,” Sep. 2018. [<a href="https://arxiv.org/abs/1809.00219">Arxiv</a>]

:arrow_right: X. Wang, L. Xie, C. Dong, and Y. Shan, “Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data,” Jul. 2021. [<a href="https://arxiv.org/abs/2107.10833">Arxiv</a>]
