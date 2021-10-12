import boto3
from json import loads, dumps

runtime = boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    body = loads(event['body'])
    csvdata = ','.join((body['nitrogen'], body['phosphorous'], body['pottasium'], body['temperature'], body['humidity'], body['ph'], body['rainfall']))
    response = runtime.invoke_endpoint(EndpointName='xgboost-2021-10-01-15-42-07-784-1', ContentType='text/csv', Body=csvdata)
    result = loads(response['Body'].read().decode())
    return {
        	"statusCode":200,
        	"headers":{
        		"Access-Control-Allow-Origin":"*",
        	},
        	"body":dumps({
        		"canGrowTomato":True if result > 0.5 else False
        	})
        }