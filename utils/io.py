import orjson

def fast_save_to_jsonl(
    dataset: dict,
    output_path: str,
    buffer_size: int = 10_000
) -> None:
    """
    Fast single-process saving of a Hugging Face Dataset to JSONL using orjson.
    
    Args:
        dataset (Dataset): The dataset to save.
        output_path (str): The output JSONL file path.
        buffer_size (int, optional): Number of items to accumulate before writing out. Defaults to 10,000.
        show_progress (bool, optional): Whether to show a progress bar. Defaults to True.
    """
    total = len(dataset)
    with open(output_path, 'ab') as f:
        iterator = dataset
        
        buffer = bytearray()
        count = 0
        
        # Predefine a dump function for speed
        dump = orjson.dumps
        newline = b"\n"
        
        for item in iterator:
            # Serialize directly to bytes using orjson
            # Note: orjson.dumps returns bytes, so we just append a newline.
            buffer += dump(item) + newline
            count += 1
            
            # When buffer is large enough, write it once to reduce I/O overhead
            if count >= buffer_size:
                f.write(buffer)
                buffer.clear()
                count = 0
        
        # Write any remaining items in buffer
        if buffer:
            f.write(buffer)