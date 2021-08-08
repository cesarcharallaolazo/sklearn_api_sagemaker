import boto3
import json
import io
import os

ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
runtime = boto3.client("runtime.sagemaker")


def lambda_handler(event, context):
    # TODO implement
    payload = event

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",
        Accept="application/json",
        Body=payload
    )
    response_body = json.loads(response["Body"].read())

    return response_body
