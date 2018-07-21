# Chinese Character Classification through CNNs

Given their ability to characterize different types of images, convolutional neural networks (CNNs) have been used extensively in the problem of handwritten character classification. Hnadwriting differs from person to person, so a computer can often struggle to match a image with a corresponding character. Interested in learning about CNNs myself, I decided to give this problem a try by training and testing on Chinese characters from the CASIA dataset (http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html).

## Data Preprocessing

The images were extracted from the CASIA dataset using an open-source library called PyCasia. They were then rescaled to 128 by 128, converted to grayscale, and randomly cropped to 96 by 96 for training. The images in the validation data set were simply rescaled to 96 by 96.

In the end, the training dataset was composed of 897760 images with 3755 classes, while the testing dataset was composed of 224000 images.

## File Roles

To run the CNN, type in the following commands (in order):
```
python3 dict.py   # creates a dictionary of the characters in the training dataset

python3 image.py  # reads the images through PyCasia and creates the training and testing tensors

python3 main.py   # builds the model
```

The best model is saved to model.pt, which can be reloaded for future use using Pytorch.

## Model Framework

Here, I implement a two-layer CNN with two max-pooling layers and one fully connected layer. In the future, I may change the a few of the parameters, but so far, this setup has achieved the highest validation accuracy.

## Results

Overall, the classifier achieved a best accuracy of 87.24% on the training dataset and 75.62% on the validation dataset. By Epoch 25, the percentages were still climbing upwards, so a little more testing is needed to determine the true best accuracies.

![results](https://raw.githubusercontent.com/williamhu99/chinese-character-classifier/master/Images/results.png)
