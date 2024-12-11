from typing import Optional
import orjson
from utils.io import fast_save_to_jsonl

def process_conversation(conv_data: dict, image_token = "<image>") -> Optional[dict]:
    """
    Helper function to convert conversation data to LLaVA format.
    """
    conversations = orjson.loads(conv_data['data']).get('conversations', [])
    # Early validation - need at least 2 messages
    if len(conversations) < 2:
        return None
        
    # Pre-allocate the result list with known size
    llava_conv = [None] * len(conversations)
    image_paths = set()
    
    # Process system message first if it exists
    system_text = ""
    start_idx = 0
    if conversations[0]['role'] == 'system':
        system_text = ' '.join(c['value'] for c in conversations[0]['content'] 
                                if c['type'] == 'text')
        start_idx = 1
    
    # Single-pass processing of messages
    conv_idx = 0
    for i in range(start_idx, len(conversations)):
        msg = conversations[i]
        
        if msg['role'] == 'user':
            # Process all content at once using list comprehension
            message_parts = []
            if system_text and conv_idx == 0:
                message_parts.append(system_text)
                
            # Single comprehension to handle both text and images
            message_parts.extend(
                image_token if c['type'] == 'image' and image_paths.add(c.get('value', '')) is None
                else c['value'] if c['type'] == 'text'
                else ''
                for c in msg['content']
            )
            
            llava_conv[conv_idx] = {
                "role": "user",
                "content": ''.join(message_parts).strip()
            }
        
        elif msg['role'] == 'assistant':
            llava_conv[conv_idx] = {
                "role": "assistant",
                "content": ''.join(c['value'] for c in msg['content'] if c['type'] == 'text')
            }
        
        conv_idx += 1
    
    # Trim any unused slots and validate
    llava_conv = [msg for msg in llava_conv[:conv_idx] if msg is not None]
    
    # Validation checks
    if (len(llava_conv) == 0 or 
            len(llava_conv) % 2 != 0 or
            not all(msg['role'] == ('user' if i % 2 == 0 else 'assistant') or
            len(msg["images"]) == len(msg["content"].split(image_token)) - 1
                for i, msg in enumerate(llava_conv))):
        return None
    
    # Create result with all valid images
    info = {
        "id": f"{conv_data['group_id']}_{conv_data['uuid']}",
        "images": list(image_paths),  # Convert set to list
        "conversation": llava_conv
    }
    return info

    
def EXPORT(batch_data, output_path):
    """
    Convert conversation data to LLaVA format and append to a JSONL file.
    Optimized for performance.
    
    Args:
        batch_data (list): List of conversation dictionaries
        output_path (str): Path to the output JSONL file
    """
    convs = [process_conversation(conv) for conv in batch_data]
    valid_convs = [conv for conv in convs if conv is not None]
    fast_save_to_jsonl(valid_convs, output_path)
    return len(valid_convs)