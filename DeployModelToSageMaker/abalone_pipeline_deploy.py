#!/usr/bin/env python
# coding: utf-8

# ## SETUP SAGEMAKER

# In[2]:


bucket


# In[1]:


import sagemaker
from sagemaker import get_execution_role

sagemaker_session = sagemaker.Session()

# Get a SageMaker-compatible role used by this Notebook Instance.
role = get_execution_role()

# S3 prefix
bucket = sagemaker_session.default_bucket()
prefix = "Scikit-LinearLearner-pipeline-abalone-example"


# In[ ]:





# ## PREPROCESSING

# #### Get Dataset

# In[4]:


get_ipython().system('wget --directory-prefix=./abalone_data https://s3-us-west-2.amazonaws.com/sparkml-mleap/data/abalone/abalone.csv')


# In[9]:


get_ipython().system('ls')


# #### Upload the data for training

# In[10]:


WORK_DIRECTORY = "abalone_data"

train_input = sagemaker_session.upload_data(
    path="{}/{}".format(WORK_DIRECTORY, "abalone.csv"),
    bucket=bucket,
    key_prefix="{}/{}".format(prefix, "train"),
)


# In[ ]:





# #### create script: sklearn_abalone_featurizer.py

# In[ ]:





# ## Create SageMaker Scikit Preprocessor

# In[11]:



from sagemaker.sklearn.estimator import SKLearn

FRAMEWORK_VERSION = "0.23-1"
script_path = "sklearn_abalone_featurizer.py"

sklearn_preprocessor = SKLearn(
    entry_point=script_path,
    role=role,
    framework_version=FRAMEWORK_VERSION,
    instance_type="ml.c4.xlarge",
    sagemaker_session=sagemaker_session,
)


# In[12]:


sklearn_preprocessor.fit({"train": train_input})


# In[ ]:





# ### Batch transform Job for our training data

# In[16]:


transformer = sklearn_preprocessor.transformer(
    instance_count=1, instance_type="ml.c4.xlarge", assemble_with="Line", accept="text/csv"
)


# In[17]:


transformer.transform(train_input, content_type="text/csv")
print("Waiting for transform job: " + transformer.latest_transform_job.job_name)
transformer.wait()
preprocessed_train = transformer.output_path


# In[18]:


preprocessed_train


# In[ ]:





# ## Fit a LinearLearner Model with the preprocessed data 

# In[19]:


import boto3
from sagemaker.image_uris import retrieve

ll_image = retrieve("linear-learner", boto3.Session().region_name)


# In[21]:


ll_image


# In[20]:


s3_ll_output_key_prefix = "ll_training_output"
s3_ll_output_location = "s3://{}/{}/{}/{}".format(
    bucket, prefix, s3_ll_output_key_prefix, "ll_model"
)

ll_estimator = sagemaker.estimator.Estimator(
    ll_image,
    role,
    instance_count=1,
    instance_type="ml.c4.xlarge",
    volume_size=20,
    max_run=3600,
    input_mode="File",
    output_path=s3_ll_output_location,
    sagemaker_session=sagemaker_session,
)

ll_estimator.set_hyperparameters(feature_dim=10, predictor_type="regressor", mini_batch_size=32)

ll_train_data = sagemaker.inputs.TrainingInput(
    preprocessed_train,
    distribution="FullyReplicated",
    content_type="text/csv",
    s3_data_type="S3Prefix",
)

data_channels = {"train": ll_train_data}
ll_estimator.fit(inputs=data_channels, logs=True)


# In[ ]:





# ## CREATE PIPELINE

# ### Setup pipeline

# In[22]:


from sagemaker.model import Model
from sagemaker.pipeline import PipelineModel
import boto3
from time import gmtime, strftime

timestamp_prefix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

scikit_learn_inferencee_model = sklearn_preprocessor.create_model()
linear_learner_model = ll_estimator.create_model()

model_name = "inference-pipeline-" + timestamp_prefix
endpoint_name = "inference-pipeline-ep-" + timestamp_prefix
sm_model = PipelineModel(
    name=model_name, role=role, models=[scikit_learn_inferencee_model, linear_learner_model]
)

sm_model.deploy(initial_instance_count=1, instance_type="ml.c4.xlarge", endpoint_name=endpoint_name)


# In[23]:


model_name


# In[24]:


endpoint_name


# In[ ]:





# ## MAKE REQUEST

# In[25]:


from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer

payload = "M, 0.44, 0.365, 0.125, 0.516, 0.2155, 0.114, 0.155"
actual_rings = 10
predictor = Predictor(
    endpoint_name=endpoint_name, sagemaker_session=sagemaker_session, serializer=CSVSerializer()
)

print(predictor.predict(payload))


# In[ ]:





# In[ ]:




