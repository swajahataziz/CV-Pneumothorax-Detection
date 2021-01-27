# SageMaker Pneumothorax PyTorch Demo

This is a demo of building a Semantic Segmentation solution for detecting Pneomothorax in PyTorch.

## Identifying Pneumothorax disease in chest x-rays using SageMaker
Amazon SageMaker is a fully managed service that provides developers and data scientists with the ability to build, train, and deploy machine learning (ML) models quickly. Amazon SageMaker removes the heavy lifting from each step of the machine learning process to make it easier to develop high-quality models. The SageMaker Python SDK makes it easy to train and deploy models in Amazon SageMaker with several different machine learning and deep learning frameworks, including PyTorch.

In this notebook, we use Amazon SageMaker to train a convolutional neural network using PyTorch and pneumothorax X-ray data, and then we host the model in Amazon SageMaker for inference.

# What is Pneumothorax
From Wikipedia:
"A pneumothorax is an abnormal collection of air in the pleural space between the lung and the chest wall. Symptoms typically include sudden onset of sharp, one-sided chest pain and shortness of breath. In a minority of cases, a one-way valve is formed by an area of damaged tissue, and the amount of air in the space between chest wall and lungs increases; this is called a tension pneumothorax. This can cause a steadily worsening oxygen shortage and low blood pressure and unless reversed can be fatal. Very rarely, both lungs may be affected by a pneumothorax. It is often called a "collapsed lung", although that term may also refer to atelectasis."

## How is the problem identified in an X-Ray?
The entities we can identify in a chest X-Ray could include:
* Air: Black spots
* Fluids & Tissues: Grey
* Bone & Solides: White

Air enclosures may possibly be just mild disturbances in the chest X-Ray. Convulutional Neural Networks are excellent at identifying abnormalities.

This exercise involves using chest X-rays and masks. The kernel used is based on UNet architecture with ResNet34 encoder. We have used segmentation_models.pytorch library which has many inbuilt segmentation architectures, details of which can be found [here](https://github.com/qubvel/segmentation_models.pytorch). 

What's down below?

    UNet with imagenet pretrained ResNet34 architecture
    Training on 512x512 sized images/masks with Standard Augmentations
    MixedLoss (weighted sum of Focal loss and dice loss)
    Gradient Accumulution


## Setup
We start by installing some required dependencies including the segmentation_models.pytorch and albumentations (used for transforming images during pre-processing)
Additionally, we will set-up:


* An Amazon S3 bucket and prefix for training and model data. This should be in the same region used for SageMaker Studio, training, and hosting.
* An IAM role for SageMaker to access to your training and model data. If you wish to use a different role than the one set up for SageMaker Studio, replace sagemaker.get_execution_role() with the appropriate IAM role or ARN. For more about using IAM roles with SageMaker, see the AWS documentation.
* Download the training data from s3:////hcls-datasets/cv-pneumothorax-detection/ and upload it into the directory to be used for model training under prefix
* Update buckets and prefix variables in cell 2
* Upload a sample of images under input/sample, input/train to be used for data exploration during workshop walkthrough.

