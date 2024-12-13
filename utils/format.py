import hashlib
import re
from typing import List, Dict, Any

def name_url(url):
    """
    Generate a unique name for a URL.
    """
    return hashlib.md5(url.encode()).hexdigest()[:8] + "-" + url.split("/")[-1].split("?")[0][-100:]

def sanitize_filename(text):
    """
    Convert text to a valid filename by:
    1. Replacing slashes and backslashes with underscores
    2. Keeping existing hyphens and underscores
    3. Keeping alphanumeric characters
    4. Removing/replacing all other special characters
    5. Trimming leading/trailing spaces and dots
    
    Args:
        text (str): Input text to sanitize
        
    Returns:
        str: Sanitized filename
    """
    # Replace slashes and backslashes with underscores
    sanitized = re.sub(r'[/\\]', '_', text)
    
    # Replace other invalid filename characters with underscores
    # This includes: < > : " | ? * and control characters
    sanitized = re.sub(r'[<>:"|?*\x00-\x1F\x7F]', '_', sanitized)
    
    # Replace multiple underscores with a single underscore
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip('. ')
    
    # If empty, provide a default name
    if not sanitized:
        sanitized = 'unnamed_file'
        
    return sanitized

def image_conversation(question, answer, images):
    """
    Args:
        question (str): Question text
        answer (str): Answer text
        images (List[str]): List of image URLs

    Returns:
    [
        {
            "role": str,
            "content": [
                {
                    "type": "text",
                    "value": str
                }
            ]
        }
    ]
    """
    image_content = [{"type": "image", "value": image} for image in images]
    return [
        {
            "role": "user", 
            "content": image_content + [
                {"type": "text", "value": question},
            ]
        },
        {
            "role": "assistant", 
            "content": [
                {"type": "text", "value": answer},
            ]
        }
    ]

def chatml_to_conversation(chatml_data: List[Dict[str, str]]) -> Dict[str, List[Dict[str, List[Dict[str, str]]]]]:
    """
    Convert ChatML format to multi-modal format.
    
    Args:
        chatml_data: A list of dictionaries in ChatML format
        
    Returns:
        Dictionary in multi-modal format
    """
    # Validate input
    if not isinstance(chatml_data, list):
        raise ValueError("Input must be a list of messages")
        
    conversations = []
    
    # Convert each message
    for message in chatml_data:
        if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
            raise ValueError("Each message must have 'role' and 'content' fields")
            
        # Validate role
        if message['role'] not in ['system', 'user', 'assistant']:
            raise ValueError("Role must be one of: system, user, assistant")
            
        # Convert to multi-modal format
        converted_message = {
            "role": message['role'],
            "content": [
                {
                    "type": "text",
                    "value": message['content']
                }
            ]
        }
        conversations.append(converted_message)
    
    # Create final structure
    result = {
        "conversations": conversations
    }
    
    return result