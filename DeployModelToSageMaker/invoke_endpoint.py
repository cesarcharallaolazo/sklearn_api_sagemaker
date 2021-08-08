#!/usr/bin/env python
# coding: utf-8

# # INVOKE ENDPOINT

# In[2]:


import boto3
import json

client = boto3.client('sagemaker-runtime')

payload = "M, 0.44, 0.365, 0.125, 0.516, 0.2155, 0.114, 0.155"
actual_rings = 10



# In[3]:


response = client.invoke_endpoint(
    EndpointName="inference-pipeline-ep-2021-08-05-19-04-05",
    ContentType="text/csv",
    Accept="application/json",
    Body=payload
)

response_body = json.loads(response['Body'].read())
print(json.dumps(response_body, indent=4))


# In[ ]:




