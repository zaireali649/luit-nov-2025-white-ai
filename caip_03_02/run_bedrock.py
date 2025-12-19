from call_bedrock import call_bedrock
import boto3
import json
import time
import random
from botocore.exceptions import ClientError

flower_data = []
# Create a Bedrock runtime client
bedrock = boto3.client('bedrock-runtime')

# read from file
with open('prompt_template.txt', 'r') as file:
    prompt = file.read()

# List of all 50 US states
states = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", 
    "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", 
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", 
    "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", 
    "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", 
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", 
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", 
    "Wisconsin", "Wyoming"
]

def call_bedrock_with_retry(bedrock_client, prompt, max_retries=3):
    """
    Call Bedrock with exponential backoff retry logic for throttling.
    """
    for attempt in range(max_retries):
        try:
            return call_bedrock(bedrock_client, prompt)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ThrottlingException':
                if attempt < max_retries - 1:  # Don't wait on last attempt
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"  Throttled! Waiting {wait_time:.1f} seconds before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                else:
                    print(f"  Max retries ({max_retries}) exceeded for this request")
                    raise
            else:
                # Re-raise non-throttling errors immediately
                raise
    return None

few_states = ["Georgia", "Florida", "Texas", "California", "New York"]

# Process each state
for i, state in enumerate(few_states):
    print(f"Processing {state}... ({i+1}/50)")
    
    # Replace the state name in the prompt
    current_prompt = prompt.replace("{state_name}", state)
    
    try:
        # Call Bedrock with retry logic
        responses = call_bedrock_with_retry(bedrock, current_prompt)
        
        # Parse and append each response
        for response in responses:
            flower_data.append(json.loads(response))
            
        print(f"Successfully processed {state}")
        
    except Exception as e:
        print(f"Failed to process {state}: {str(e)}")
        continue
    
    # Add a small delay between requests to avoid throttling
    if i < len(states) - 1:  # Don't wait after the last state
        time.sleep(1)  # 1 second delay between requests

with open('flower_data.json', 'w') as f:
    json.dump(flower_data, f, indent=4)

print("Done")