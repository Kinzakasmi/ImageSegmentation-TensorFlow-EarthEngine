# **Landcover classification using TensorFlow and Google Earth Engine** #
Welcome to the landcover classification project.
This projects aims at: 
1. Creating a kml file of a mask. This mask is an image where every color represents a certain class:
	- Lime green for fields
	- Dark green for forests
	- Magenta for urban areas
	- Blue for water
2. Assigning a class to each pipe of a network (could be anaything, like a water network) according to its location (the result is a CSV file).

There are two jupyter-notebook tutorials that guide you through the training and prediction processes. 
A jupyter notebook is a web-based interactive computational environment that is dynamic and easy to use. 
You can read the notebooks using anaconda or vs-code after installing the jupyter-kernel.

- The notebook **tuto_deep_learning.ipynb** is a notebook that uses deep learning and creates a kml-mask file.
- The notebook **tuto_machine_learning.ipynb** uses machine learning algorithms. 

*If you already have a pretrained model in your file "models" and want to use it to predict some area of you network, please follow the instructions and do not execute the training section. This will be reminded in the notebook.*

The `utils` folder contains scripts defining:
- Utility functions,
- Learning rate schedulers,
- Losses and metrics.

The `dataset` folder contains scripts for:
- Dataset construction: definition of a TFDatasetConstruction class,
- Dataset loading and preprocessing: definition of a TFDatasetProcessing class and NPDatasetProcessing class.

The `models` folder contains the following scripts:
- models.py: defines a ModelTrainingAndEvaluation class whose methods allow :
	- Training and finetuning of KNN, SVM and RandomForest models,
	- Model evaluation; score and confusion matrix calculation,
	- PCA and UMAP dimentionaly reduction,
	- Feature importance computing in case of RandomForest.
- unet.py: 
	- Defines a DLModel class whose methods allow UNET model implementation and initialisation from a checkpoint (you could add your own model as a method),
	- ModelEvaluation; score and confusion matrix calculation.

Good prediction.
