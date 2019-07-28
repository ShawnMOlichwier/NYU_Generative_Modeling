# Generative modeling research
Research conducted at the Tandon School of Engineering at NYU during the summer of 2019.

Building deep models using autoencoder architecture. The actual nuts and bolts of each model will vary. 

# Abstract for the project:
The area of generative machine learning has made great improvements in recent years. Models are able to create realistic reconstructions of various types of images. Many of these neural networks construct new images by creating a new pixel representation, a bitmap image.  Our method, inspired by human artistic methods, recreate images digitally by generating virtual “paint strokes” as represented by Scalable Vector Graphic (SVG) instructions. Vector graphics store data as a set of mathematical instructions which are interpreted by graphics software to create an image. An image in vectorized format lends itself well for data compression, which has implications for smaller data storage and transfer volume.

This project is an analysis with various combinations of deep learning models. We focus on an autoencoder which takes a rasterized image as input. Once compressed to a lower dimensional latent space, we experiment with ways to recreate the image. All of our methods give SVG <path> parameters as output. Lastly, our images is created from these parameters and our loss can be computed for backpropagation.


# The dataset:
The dataset used is the Large-scale CelebFaces Attributes (CelebA) Dataset. The images all share similar structure so it feels like a good starting point for our model to try to learn. The images needed to be square (128, 128), which was taken care of when the data is loaded.

The data can be found here:
http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html



