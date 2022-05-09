# Dynamic-kernels
This repository contains the code for the paper: <br>
"Context-aware machine learning with dynamically assembled weight matrices."
*link coming soon*

# Code
All the code is written in the newest stable tensorflow version v2.5.0. <br>
The code is designed with to be compatible with the keras functional, sequantial & model API.

# Project tree
* Src
  * Optimizers
  * Layers
  * Models
  * Callbacks
  * Initializers
* Notebooks

# Example
The following is a sample for code usage.

```
from src.layers import CAAddDense
from src.optimizers import SVDAdam
from src.models.utils import wrap_model

# Create a dataset
data = tf.data.Dataset.from_generator(...)
# Make a context model
context_inputs = tf.keras.layers.Inputs(...)
context_outputs = tf.keras.layers.Dense(...)(context_inputs)
context_model = tf.keras.models.Model()(outputs=context_outputs, inputs=context_inputs)
# Create a model
inputs = tf.keras.layers.Inputs(...)
context = tf.keras.layers.Inputs(...)
context = context_model(context)
hidden = SVDDense(...)([inputs, context])
... # Add more complicated architecture
outputs = tf.keras.layers.Dense(...)(hidden)
# Make model
model = tf.keras.Models(outputs=outputs, inputs=(inputs, context))
# Wrap model such that dataset is given to optimizer
model = wrap_model(model)
# Create optimizer --> Needs a model and optional context model to be created
optimizer = SVDAdam(model, context_model, ...)
# Create loss object
loss_fn = tf.keras.losses.MeanSquaredError()
# Compile model
Model.compile(optimizer, loss_fn, ...)
# train model
model.fit(data, ...)
```

More detailed examples, corresponding to the experiment section in the paper, can be found in the notebooks.

The following notebooks are available:
* Artificial2D contains a two dimensional example for the artificial data.
* ArtificialMultiDim contains higher dimensional modeling and analysis for the artificial data.
* Indian Covid Data - Baseline contains time series analysis and modeling for the per state individual models.
* Indian Covid Data contains time series analysis and modeling corresponding to the general models.
* DataAnalysis contains the code used for scraping of the image tags for the YFCCM100-GEO100 dataset.
* GeoImagesModel contains models and analysis used for the the YFCCM100-GEO100 dataset.


# Data

Data corresponding to the indian covid time series: *Coming soon* <br>
Data corresponding to the YFCCM100-GEO100 and tags data: *Coming soon* <br>

