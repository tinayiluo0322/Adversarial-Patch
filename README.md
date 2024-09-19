# Adversarial-Patch

##Notebook Summary: Adversarial Patch Creation and Evaluation##

This notebook presents the development, training, and evaluation of adversarial patches designed to manipulate the output of a pretrained ResNet-34 model on the ImageNet dataset. The goal of these patches is to force the model to misclassify images into a target class ("broccoli") by applying the patch to images. Throughout the notebook, I explore various methods to create effective adversarial patches, including versions with and without visual disguises, while ensuring robustness against random noise and image transformations.

### Key Components of the Notebook:

1. **Dataset Preparation and Model Setup**:
   The notebook begins by setting up the environment, loading the ResNet-34 model pretrained on ImageNet, and preparing the TinyImageNet dataset for training and validation. The model is evaluated to ensure it can classify images correctly before introducing adversarial patches.

2. **Adversarial Patch Generation**:
   The core functionality of the notebook revolves around the generation of adversarial patches. The patches are trained by minimizing the cross-entropy loss between the model’s predictions and a specified target class ("broccoli"). Each patch is initialized as a learnable parameter and placed at random locations on training images, with Gaussian noise added to increase robustness.

3. **Patch Training and L-Infinity Constraint**:
   The patches are trained using Stochastic Gradient Descent (SGD) with momentum over several epochs. For disguised versions of the patch, an L-infinity norm constraint is introduced, ensuring the pixel values stay within a certain range to prevent the patch from becoming visually conspicuous. Additionally, various versions of the patch, including disguised and non-disguised options, are created and trained separately.

4. **Evaluation and Metrics**:
   After training, the patch is evaluated on a validation set by applying it to random locations on images and measuring its ability to mislead the classifier into predicting "broccoli." The notebook records both top-1 and top-5 accuracy to assess the patch's effectiveness. Key metrics, such as average loss per epoch, final accuracy, and top-5 accuracy, are saved for each version of the patch.

5. **Saving and Visualization**:
   Each trained patch is saved in multiple formats, including a `.pt` model file for further experimentation, a `.png` image for visualization, and a JSON file containing the training metrics. The patch images are visualized at the end of the training process, allowing users to inspect the patch design.

### Results and Conclusion

**Adversarial Patch Version 1: A Non-Disguised Broccoli**

The first version of the adversarial patch is designed to manipulate the predictions of a ResNet-34 image classifier by forcing it to classify any image as "broccoli" when the patch is applied. This version focuses purely on adversarial effectiveness without any attempt to visually disguise the patch. The patch is trained by placing it at random offsets on images, using a ResNet-34 model pretrained on ImageNet. To enhance robustness, random Gaussian noise is added to the images during each training iteration.

The patch is optimized by minimizing the cross-entropy loss between the model’s predictions and the target class "broccoli." Over the course of several epochs, the patch parameters are updated using Stochastic Gradient Descent (SGD) with momentum. The goal is to craft a pattern that, when placed on any image, causes the classifier to output "broccoli" as the top prediction, regardless of the image's actual content.

The training process showed consistent improvement, with the following losses recorded over five epochs:
- **Epoch 0**: Average loss = 1.3785
- **Epoch 1**: Average loss = 0.0561
- **Epoch 2**: Average loss = 0.0340
- **Epoch 3**: Average loss = 0.0308
- **Epoch 4**: Average loss = 0.0246

After training, the patch was evaluated by randomly applying it to images in the validation set. The model's performance was measured using top-1 and top-5 accuracy metrics:
- **Final Top-1 Accuracy**: 0.9386 (93.86% of images were classified as "broccoli")
- **Final Top-5 Accuracy**: 0.9809 (98.09% of images included "broccoli" in the top 5 predictions)

The patch was saved in multiple formats, including a `.pt` file for further experimentation and a `.png` image for visualization. All training metrics, including loss per epoch and accuracy results, were saved in a JSON file for reproducibility. This non-disguised patch effectively increases the classifier’s likelihood of predicting "broccoli" when applied to any image.

**Adversarial Patch Version 2: A Disguised Broccoli**

In this version of the adversarial patch, the objective is to create a visually disguised patch that still manipulates the classifier into predicting "broccoli" as the label for any input image. The patch is constrained by an L-infinity norm, which ensures that the pixel values stay within a limited range, making the patch subtle and less noticeable. The model is trained using cross-entropy loss, which maximizes the probability of predicting "broccoli" when the patch is applied to an image.

The patch is optimized through several epochs using Stochastic Gradient Descent (SGD), and random Gaussian noise is added to the images during training to improve robustness against noise. After each backpropagation step, the patch is projected back into the L-infinity constraint to maintain its subtle appearance. The patch is applied to different locations on the image, forcing the classifier to misclassify the input as "broccoli."

The training process over five epochs yielded the following metrics:
- **Epoch 0**: Average loss = 1.6360
- **Epoch 1**: Average loss = 0.1048
- **Epoch 2**: Average loss = 0.0808
- **Epoch 3**: Average loss = 0.0774
- **Epoch 4**: Average loss = 0.0703

After training, the patch was evaluated on a validation set by placing it randomly on the images. The patch demonstrated strong performance, achieving:
- **Final Top-1 Accuracy**: 0.8968 (89.68% of images were classified as "broccoli")
- **Final Top-5 Accuracy**: 0.9546 (95.46% of images included "broccoli" in the top 5 predictions)

The trained patch was saved as a `.pt` file and visualized as a `.png` image. All training metrics, including the loss per epoch, accuracy, and top-5 accuracy, were saved in a JSON file for further analysis. The result is a highly effective, visually disguised adversarial patch that forces the classifier to predict "broccoli" without being overtly detectable.
