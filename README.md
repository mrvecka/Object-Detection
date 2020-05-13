# Object-Detection-Diploma

Our job is to determine the position of the object in image and draw bounding box around that object. We use KITTI dataset to train and test designed CNN.
We decided to detect only cars so dataset labels were adapted to this purpose. Highly ocluded were removed and also object which part was outside of image was removed.
Program will automaticly create this upgraded labels and store them to different folder. On second run program use previously created files which contains only relevant objects.
this is how the training dataset is created. During training data are randomly sampled. Network is designed to detect object of diferent size in one pass.
That is why network is divided to 4 parts. Each part has it's own loss function and optimizer which optimize only variables which belong to that part.
This project provide implementation in older 1.14 and newer 2.1 version of TensorFlow framework. Results are nearly same, but thera are hude difference in speed. 


## Object detection with TF 1.14

At first we use Tensorflow framework version 1.15. It was sufficient because of our experience in this version. We did a lot of bugfixing and optimalization.
At the end network was able to detect objects with quite high precision. This version suffer from speed. 50 iteration with batch_size took 40s.
Code is under master branch of this project.

### Prerequisites

TensorFlow 1.14.0  
TensorBoard 1.14.0  
opencv-python 4.2.0.34  
numpy 1.18.3  
Python 3.6.7  

## Object detection with TF 2.1.0

We decided to upgrade packages and also framework version. We use TensorFlow version 2.1  which provide full keras support and also OOP support,
We take that advantage and recreate network and loss function using abstract classes from tf.keras moduls.
Despite all changes network architecture remain same as it was in Tf 1.14. We also use advantage of TF 2.1 function called @tf.function which speed up network 
by around 80%. 50 iterations with batch_size took only 6s. Code is under tf_2.0 branch of this project.

**Highly recommended**

### Prerequisites

TensorFlow 2.1.0  
TensorBoard 2.1.1  
TensorFlow-GPU 2.1.0  
tensorflow-probability 0.9.0  
opencv-python 4.2.0.34  
numpy 1.18.3  
Python 3.6.7  
CUDA 10.1  
cudnn 7.6.5  

## Results

Results of both implementations are quite same so we will post it oonly once. White boxes represents bounding bosex from dataset. Colored bounding boxes are results of neural network. This boxes are oriented. 
Red part of box determine a front side of car blue one rear side of car. This provide additional information of object ie car direction.
![Scale 2 - smallest objects](https://github.com/mrvecka/Object-Detection-Diploma/blob/tf_2.0/output/output_s2.jpg)
![Scale 4](https://github.com/mrvecka/Object-Detection-Diploma/blob/tf_2.0/output/output_s4.jpg)
![Scale 8](https://github.com/mrvecka/Object-Detection-Diploma/blob/tf_2.0/output/output_s8.jpg)
![Scale 2 - largest objects](https://github.com/mrvecka/Object-Detection-Diploma/blob/tf_2.0/output/output_s16.jpg)


## Network settings

To make program more scalable and get rid of editing files directly we use python config file which provide all neccessary settings for loading, training, testing and saving model.
There are lot of settings which could be set but i will not cover them all and recommend leave them as they are.

### Minimum training settings
**IMAGE_PATH** - path to images from dataset  
**CALIB_PATH** - path to calibration files  
**LABEL_PATH** - path to label files from dataset (KITTI format)  
**BB3_FOLDER** - path to upgraded label files (they are created during first run)
	if path is empty files are not create, if you want to just create this files run .\Services\loader.py  
**IMG_WIDTH** - width of network input image (images are scaled during loading)  
**IMG_HEIGHT** - height of network input image  
**IMG_CHANNELS** - number of channel of image, allowed value 1(grayscale) or 3(colored)  
**SAVE_MODEL_EVERY** - model weights will be saved every N epochs  
**ITERATIONS** - count of iterations per epoch  
**BATCH_SIZE** - size of batch  
**LEARNING_RATE** - learning rate of optimizer  
**UPDATE_LEARNING_RATE** - number of epoch when learning rate should be updated (lowered) ie [100, 200, 800]  
**OPTIMIZER** - type of optimizer, alowed values "adam" or "sgd"  
**DATA_AMOUNT** - determine the size of dataset which should be used for training, -1 for whole dataset  
**SPECIFIC_DATA** - train network on one image, file name should be provided  

### Minimum testing settings
**SPECIFIC_TEST_DATA** - test network on one image, file name should be provided ie "000008" (KITTI)  
**RESULT_TRESHOLD** - result with probability lower than this value will be ignored and will not be shown  


After setting up run train.py or test.py.

	
	




