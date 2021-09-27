# **Landcover classification using TensorFlow and Google Earth Engine ** #
Welcome to the landcover classification project.
This projects aims at : 
1. Creating a kml file of a mask. This mask is an image where every color represents a certain class :
	Lime green for fields
	Dark green for forests
	Magenta for urban areas
	Blue for water
2. Assigning a class to each pipe of a network according to its location.

There are two jupyter-notebook tutorials that guide you through the training and prediction processes. 
A jupyter notebook is a web-based interactive computational environment that is dynamic and easy to use. 
You can read the notebooks using anaconda or vs-code after installing the jupyter-kernel.

The notebook **tuto_deep_learning.ipynb** is a notebook that uses deep learning and creates a kml-mask file.

The notebook **tuto_machine_learning.ipynb** uses machine learning algorithms.
If you already have a pretrained model in your file "models" and want to use it to predict some area of you network, please follow the instructions and do not execute the training section. This will be reminded in the notebook.

Good prediction.
