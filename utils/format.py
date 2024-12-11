import hashlib
import re

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