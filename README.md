# Face-Recognition

This repository is a face recognition platform with [TensorFlow](https://www.tensorflow.org/).

It use [CASIA](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) as training set and [LFW](http://vis-www.cs.umass.edu/lfw/) as test set.

[Docker](https://www.docker.com/) is recommended.

### Code Structure

Here is the main code structure of the repository.

Attention: the code is under TensorFlow 1.1 and python 2.7.

```
// Directories
├── dataset/			// Datasets, including CASIA, LFW
├── tfrecord/			// TFRecord for tensorflow
├── images/				// Output images
├── outdated/			// Some old code
├── tools/				// Some tool files
├── train_data/			// Store train data
├── features/			// Store features when test
└── txt/				// Text files for data

// Main files
├── make_tfrecords.py	// Make images into one tfrecord file
├── multi_gpu_train.py	// Main training file for multi-gpu train
├── trans_config.py		// Config file for transform
├── transform.py		// Image transform using cv2
├── pca_fuse.py			// Feature fuse and PCA
└── test_lfw.py			// Test LFW using searching threshold

// Tool/outdated files
// You can find some ideas or here.
├── input.py			
├── pca.py
├── vgg.py
├── fuse_feature.py
├── filter_file.py
└── filter_lfw.py

// Test files
// These files are for exporting images
├── test_align.py				
└── test_tfrecord_image.py
```

### LICENSE

[MIT LICENSE](https://github.com/irmowan/Face-Recognition/blob/master/LICENSE)

### Author

[irmowan](https://github.com/irmowan)

