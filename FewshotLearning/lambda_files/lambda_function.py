import json
import logging

# from aws_lambda_powertools.event_handler import APIGatewayRestResolver

from BedrockPipeline import BedrockPipeline
from construct_dataset import load_data
from parser import parse_input

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# app = APIGatewayRestResolver()

dataset = load_data()
logger.info("Data loaded")

bedrock_pipeline = BedrockPipeline(
    dataset=dataset, 
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0")
logger.info("Pipeline started")

# @app.post("/inference")
def lambda_handler(event, context):

    logger.info("Received event: " + json.dumps(event, indent=2))

    input_text = None

    try:
        if 'body' in event:
            try:
                body = json.loads(event['body'])
                input_json = body.get('input')
                input_text = parse_input(input_json)

            except json.JSONDecodeError:
                logger.error("Failed to parse request body as JSON")
            
        # If no input found in body, try to find it directly in the event
        if not input_text:
            input_json = event.get('input')
            input_text = parse_input(input_json)

        logger.info(f"Received input: {input_text}")

        if not input_text:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Missing input parameter'})
            }
    
        llm_response = bedrock_pipeline.message_api_inference(input_text)

        logger.info('LLM called successfully')

        # Format response based on whether it's from Lambda URL or direct
        if 'body' in event:
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Headers': '*',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'OPTIONS,POST,GET',
                    'Content-Type': 'application/json'
                },
                    'body': json.dumps({ "Answer": llm_response['content'][0]['text'] })
        }
        else:  # Direct Lambda invocation
            return { "Answer": llm_response['content'][0]['text'] }
    
    except Exception as e:
        logger.exception(f"Error: {str(e)}")
        error_response = {'error': str(e)}

        if 'body' in event:
            return {
                'statusCode': 500,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps(error_response)
            }
        else:  # Direct Lambda invocation
            return error_response
        