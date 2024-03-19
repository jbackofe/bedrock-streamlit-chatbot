import json
from botocore.exceptions import ClientError

def get_secret(client, secret_name):
    """
    Retrieves a secret from AWS Secrets Manager.

    Parameters:
    - client: A boto3 client configured for Secrets Manager.
    - secret_name: The name of the secret to retrieve.

    Returns:
    A dictionary containing the secret's key-value pairs.

    Raises:
    - ClientError: An error from AWS indicating problems like network issues,
                   incorrect permissions, or a missing secret.
    - ValueError: An error indicating the secret could not be decoded.
    """
    try:
        # Attempt to retrieve the secret from AWS Secrets Manager
        response = client.get_secret_value(SecretId=secret_name)
    except ClientError as error:
        raise error

    # Parse and return the secret
    if 'SecretString' in response:
        try:
            # Attempt to parse the secret string as JSON
            return json.loads(response['SecretString'])
        except json.JSONDecodeError:
            raise ValueError("Failed to decode JSON from secret string.")
    else:
        raise ValueError("Secret does not contain a 'SecretString' field.")