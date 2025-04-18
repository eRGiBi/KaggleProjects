import json
import boto3
import random

def format_row_to_feed(inp) -> str:
    """Formats a dictionary into a string to be given to a prompt."""
    return f'Name: {inp["product_name"]}, Brand: {inp["brand"]}, \
        Retail Price: {inp["retail_price"]}, Discounted Price: {inp["discounted_price"]}, \
        Specifications: {inp["product_specifications"]}.'


class BedrockPipeline():
    """
    Based on AWS Bedrock "documentation".
    https://docs.aws.amazon.com/bedrock/latest/userguide
    """
    
    def __init__(self, 
                 dataset,
                 model_id,
                 num_examples=3,
                 model_class="sonnet",
                 max_output_token=200, temperature=0.1, top_p=0.9,
                 seed=476):

        self.model_id = model_id
        self.model_class = model_class
        self.region = "us-east-1"

        self.max_output_token = max_output_token
        self.temperature = temperature
        self.top_p = top_p

        self.num_examples = num_examples

        self.dataset = dataset
        
        random.seed(seed)

        # Lazy initialization
        self._client = None

    @property
    def client(self):
        """Lazy initialization of the boto3 client"""
        if self._client is None:
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def sample_examples(self, k) -> list:
        """Sample k number of rows from the dataset 
        and return them formatted as a complete prompt."""
        if k >= len(self.dataset):
            sampled_rows = self.dataset.copy()

        else:
            indices = random.sample(range(len(self.dataset)), k)
            sampled_rows = [self.dataset[i] for i in indices]
        
        formatted_examples = []
        for row in sampled_rows:
            formatted_examples.append({
                "Input": format_row_to_feed(row),
                "Output": row["description"]
            })
        
        return formatted_examples

    def get_sonnet_request(self, user_input):
        """Construct request with the few shot examples for inferencing Sonnet."""

        system_message = "You are an AI assistant trained to generate product descriptions based on examples. Given the following examples and a new input, you must respond in exactly the same manner, matching the examples provided."
        examples = self.sample_examples(self.num_examples)
        prompt_text = f"Examples: {examples}\nThe following is the description of the product, from which you have to generate the description: {user_input}"
        
        request = {
            "anthropic_version": "bedrock-2023-05-31", 
            "anthropic_beta": ["computer-use-2024-10-22"],
            "max_tokens": self.max_output_token,
            "system": system_message,    
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ],
            "temperature": self.temperature,
            "top_p": self.top_p
        }
        return request

    def message_api_inference(self, user_input):
        """Inferences models using Messages API"""
        request = self.get_sonnet_request(user_input)
        
        body = json.dumps(request)
        response = self.client.invoke_model(body=body, modelId=self.model_id)
        
        return json.loads(response.get('body').read())
    