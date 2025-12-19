import boto3
import json
from typing import List, Dict, Any

def construct_body(prompt: str, max_tokens: int = 2000) -> Dict[str, Any]:
    """
    Construct the request body for the Bedrock model.

    Args:
        prompt (str): The input prompt to send to the model.
        max_tokens (int, optional): Maximum number of tokens in the response. Defaults to 2000.

    Returns:
        Dict[str, Any]: The formatted request body.
    """
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": f"""Human: {prompt}"""
            }
        ]
    }

    return body

def call_bedrock(
    bedrock_client: boto3.client, 
    prompt: str, 
    max_tokens: int = 2000, 
    modelId: str = 'anthropic.claude-3-sonnet-20240229-v1:0'
) -> List[str]:
    """
    Call the Bedrock runtime with a given prompt and return model responses.

    Args:
        bedrock_client (boto3.client): A boto3 Bedrock runtime client.
        prompt (str): The prompt string to send to the model.
        max_tokens (int, optional): Maximum number of tokens in the response. Defaults to 2000.
        modelId (str, optional): The Bedrock model ID to use. Defaults to Claude 3 Sonnet.

    Returns:
        List[str]: A list of text responses from the model.
    """
    # Create the request body using the provided prompt and token limit
    body = construct_body(prompt, max_tokens=max_tokens)

    # Invoke the model with the request body
    response = bedrock_client.invoke_model(
        body=json.dumps(body),
        modelId=modelId,
    )

    # Parse the JSON response body
    result = json.loads(response["body"].read())

    # Extract and return text content from the result
    responses = [content["text"] for content in result["content"]]

    return responses

if __name__ == "__main__":
    # Create a Bedrock runtime client
    bedrock = boto3.client('bedrock-runtime')

    # Sample prompt to test the model
    prompt = "I need a few sentences on why teal is the best color.  Give only one sentence.  Be concise and to the point."

    # Call Bedrock and print responses
    responses = call_bedrock(bedrock, prompt, max_tokens=1000)

    for response in responses:
        print(response)
