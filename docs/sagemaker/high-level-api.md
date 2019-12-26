---
layout: default
title: High Level API
parent: Amazon Sagemaker
nav_order: 1
---

# High Level API
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Libraries
Sagemaker [Session object](https://sagemaker.readthedocs.io/en/stable/session.html) provides access to useful Amazon Sagemaker functions such as data upload to S3, train and transform jobs creation. It also contain attributes such as the instance's region name.

~~~ python
%matplotlib inline

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import sklearn.model_selection

import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.tuner import IntegerParameter, ContinuousParameter, HyperparameterTuner
from sagemaker.predictor import csv_serializer


session = sagemaker.Session()

# Represents the IAM role assigned when creating sagemaker notebook. 
# Allows training role to have read and write access to S3 buckets.
role = get_execution_role()
~~~

## Data Preparation
Train, validation and test data have to be uploaded to Amazon S3 for model training and inference. Amazon's built-in machine learning algorithms also requires
- removal of header and index
- target variable to be the first entry of each row.

~~~ python
# Get Data
boston = load_boston()
X_bos_pd = pd.DataFrame(boston.data, columns=boston.feature_names)
Y_bos_pd = pd.DataFrame(boston.target)

# Create Train, Validation and Test Sets
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
      X_bos_pd, Y_bos_pd, test_size=0.33
)
X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(
      X_train, Y_train, test_size=0.33
)

# Save data locally
data_dir = '../data/boston'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
X_test.to_csv(
      os.path.join(data_dir, 'test.csv'), 
      header=False, index=False
)  # remove header and index
pd.concat([Y_val, X_val], axis=1).to_csv(
      os.path.join(data_dir, 'validation.csv'), 
      header=False, index=False
)  
pd.concat([Y_train, X_train], axis=1).to_csv(
      os.path.join(data_dir, 'train.csv'),
      header=False, index=False
)  # make first column target variable

# Upload to S3
prefix = 'boston-xgboost-HL'  # S3 folder name
test_location = session.upload_data(
      os.path.join(data_dir, 'test.csv'), key_prefix=prefix
)
val_location = session.upload_data(
      os.path.join(data_dir, 'validation.csv'), key_prefix=prefix
)
train_location = session.upload_data(
      os.path.join(data_dir, 'train.csv'), key_prefix=prefix
)
~~~

## Estimator Creation and Hyperparameter Tuning
Sagemaker provide several [built-in](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html) machine learning algorithms for a variety of problems. This example utilises a built-in xgboost estimator for a regression problem.

We are also able to utilise machine learning frameworks such as [Tensorflow](https://docs.aws.amazon.com/sagemaker/latest/dg/tf.html), [Pytorch](https://docs.aws.amazon.com/sagemaker/latest/dg/pytorch.html) or our [own](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms.html) algorithms for model development.

The metric used for tuning built-in algorithms is available under the model tuning section of the selected algorithm. An example of metric table and tunable hyperparameter for xgboost is available [here](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost-tuning.html).

The best model is accessible using the tuner object's *best_training_job()* and an estimator object can be created using the attach method.

~~~ python

# Get image of algorithm for our session region
container = get_image_uri(session.boto_region_name, 'xgboost')

# Construct the estimator object.
xgb = sagemaker.estimator.Estimator(
      container, # image name of the training container
      role,      # current IAM role
      train_instance_count=1, # Number of instances to use for training
      train_instance_type='ml.m4.xlarge', 
      output_path=f's3://{session.default_bucket()}/{prefix}/output', # path for model artifacts
      sagemaker_session=session  # current sagemaker session object
)

xgb.set_hyperparameters(
      max_depth=5, eta=0.2, gamma=4,
      min_child_weight=6, subsample=0.8,
      objective='reg:linear',
      early_stopping_rounds=10,
      num_round=200
)

# Hyperparameter tuning
xgb_hyperparameter_tuner = HyperparameterTuner(
      estimator = xgb,  # estimator for the training jobs
      objective_metric_name = 'validation:rmse', # metric to compare models
      objective_type = 'Minimize', # Whether to minimize or maximize the metric.
      max_jobs = 20, # Total number of models to train
      max_parallel_jobs = 3, # Number of models to train in parallel
      hyperparameter_ranges = {
          'max_depth': IntegerParameter(3, 12),
          'eta'      : ContinuousParameter(0.05, 0.5),
          'min_child_weight': IntegerParameter(2, 8),
          'subsample': ContinuousParameter(0.5, 0.9),
          'gamma': ContinuousParameter(0, 10),
      }
)

# Wrapper for location of train and validation data, 
# to inform SageMaker data is in csv format.
s3_input_train = sagemaker.s3_input(s3_data=train_location, content_type='csv')
s3_input_validation = sagemaker.s3_input(s3_data=val_location, content_type='csv')

xgb_hyperparameter_tuner.fit({'train': s3_input_train, 'validation': s3_input_validation})
xgb_hyperparameter_tuner.wait()

# Create Estimator from best training job
xgb_attached = sagemaker.estimator.Estimator.attach(
      xgb_hyperparameter_tuner.best_training_job()
)
~~~

## Model Inference
We can then create a transformer object from the fitted estimator to perform inference on test data using the transform method. We will need to specify the format of the test data and each observation is separated. Details of transform parameters are available [here](https://sagemaker.readthedocs.io/en/stable/transformer.html#sagemaker.transformer.Transformer.transform).

~~~ python
# Batch Transform Job
xgb_transformer = xgb_attached.transformer(
      instance_count = 1,
      instance_type = 'ml.m4.xlarge'
)
xgb_transformer.transform(
      test_location, 
      content_type='text/csv', 
      split_type='Line'
)

xgb_transformer.wait()  # Print progress status in notebook

# Copy prediction to local directory
!aws s3 cp --recursive $xgb_transformer.output_path $data_dir
Y_pred = pd.read_csv(os.path.join(data_dir, 'test.csv.out'), header=None)
# Visualize Actual vs Prediction
plt.scatter(Y_test, Y_pred)
plt.xlabel("Median Price")
plt.ylabel("Predicted Price")
plt.title("Median Price vs Predicted Price")
~~~

## Model Deployment
We can also launch an predictor endpoint easily using the high level deploy method of an fitted estimator. We then define the input format type and serializer before use.

**Note: Endpoints are charged based on deployed time so remember to delete them via code or in the sagemaker console when no longer required.**
 
~~~ python
xgb_predictor = xgb_attached.deploy(
      initial_instance_count=1, 
      instance_type='ml.m4.xlarge'
)

# Set input format at endpoint
xgb_predictor.content_type = 'text/csv'
xgb_predictor.serializer = csv_serializer

Y_pred = xgb_predictor.predict(X_test.values).decode('utf-8')
# predictions is currently a comma delimited string and so we would like to break it up
# as a numpy array.
Y_pred = np.fromstring(Y_pred, sep=',')

# Terminate EndPoint
xgb_predictor.delete_endpoint()
~~~
