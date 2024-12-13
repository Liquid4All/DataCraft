from concurrent.futures import ThreadPoolExecutor, wait
from collections import deque, OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Optional, List, Deque, Dict, Union, Any, Set
import requests
import time
import os
import shutil
import pickle
from PIL import Image
import cv2
import numpy as np
import torch
import soundfile as sf
import decord
import numpy.typing as npt
from pathlib import Path


@dataclass
class FailedOperation:
    # More generic failure tracking
    source: str  # URL for downloads, file path for local operations
    target_path: str
    operation_type: str  # 'download', 'write', 'save_tensor', etc.
    id: Optional[str]
    error: str
    timestamp: float

@dataclass
class OperationStats:
    completed: int
    previously_completed: int
    pending: int
    failed: int
    failed_by_type: Dict[str, int]  # Tracks failures by operation type
    success_rate: float
    avg_file_size: float
    elapsed_time: float

def download_file(url, path, headers=None, max_retries=1, timeout=0.5, buffer_size=262144):
    default_headers = {
        'User-Agent': 'Googlebot-Image/1.0',
        'X-Forwarded-For': '64.18.15.200'
    }
    headers = headers or default_headers
    
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Get file extension from path
    file_extension = os.path.splitext(path)[1].lower()
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=timeout)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            
            # Check for content type mismatch
            if ('text/html' in content_type and 
                file_extension not in ['.txt', '.html', '.htm']):
                return {
                    'status': response.status_code,
                    'file': None,
                    'error': f'Content type mismatch: received {content_type} for file with extension {file_extension}',
                    'success': False,
                    'operation_type': 'download'
                }
            
            response.raw.decode_content = True
            bytes_written = 0
            with open(path, 'wb') as f:
                bytes_written = shutil.copyfileobj(response.raw, f, length=buffer_size)
                    
            return {
                'status': response.status_code,
                'file': path,
                'size': bytes_written,
                'content_type': content_type,
                'success': True,
                'operation_type': 'download'
            }
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                return {
                    'status': getattr(e.response, 'status_code', 408),
                    'file': None,
                    'error': str(e),
                    'success': False,
                    'operation_type': 'download'
                }
            time.sleep(2 ** attempt)

def detect_content_type(data: Any) -> str:
    """
    Detect the content type of the input data.
    Ordered by check speed (fastest first) and frequency of use.
    """
    if isinstance(data, str):
        if data.lower().startswith(('http://', 'https://', 'ftp://')):
            return 'url'
        return 'text/plain'
    
    if isinstance(data, bytes):
        return 'application/octet-stream'
    
    if isinstance(data, Image.Image):
        return 'image/pil'
    
    if torch.is_tensor(data):
        return 'tensor/pytorch'
    
    if isinstance(data, np.ndarray):
        if 2 <= data.ndim <= 3:
            if data.ndim == 2 or (data.ndim == 3 and data.shape[2] in [1, 3, 4]):
                return 'image/numpy'
        
        if data.ndim == 1 or (data.ndim == 2 and data.shape[1] in [1, 2]):
            return 'audio/numpy'
        
        return 'array/numpy'
    
    return 'unknown'

class FileManager:
    def __init__(self, num_workers=200, max_failed_history=1_000_000, ema_alpha=0.1):
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.futures = {}
        self.failed_operations: Deque[FailedOperation] = deque(maxlen=max_failed_history)
        self.lock = Lock()
        self.start_time = time.time()

        # Directory content cache
        self.directory_contents: Dict[str, Set[str]] = {}
        
        # Running statistics
        self.completed = 0
        self.failed = 0
        self.failed_by_type = {}  # Track failures by operation type
        self.ema_size = 0
        self.ema_alpha = ema_alpha
        self.previous_completed = 0

        self.currently_processing = set()

    def submit(self, data: Union[str, bytes, object], path: str, id: Optional[str] = None) -> None:
        future = self.executor.submit(self.save_file, data, path)
        with self.lock:
            self.futures[future] = {'data': data, 'path': path, 'id': id}
        future.add_done_callback(self._task_complete)

    def _ensure_directory(self, directory: Path) -> None:
        """Create directory if needed and initialize its cache"""
        directory_str = str(directory)
        if directory_str not in self.directory_contents:
            directory.mkdir(parents=True, exist_ok=True)
            self.directory_contents[directory_str] = set(os.listdir(directory_str))

    def file_exists(self, path: Path) -> bool:
        """Check if a file exists using the directory cache"""
        with self.lock:
            self._ensure_directory(path.parent)
            return str(path.name) in self.directory_contents[str(path.parent)]

    def save_file(
        self,
        data: Union[str, bytes, object],
        path: str,
        force_format: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Save data to a file with directory content caching"""
        try:
            # Avoid Race Condition
            if path in self.currently_processing:
                return {
                    'status': 200,
                    'file': str(path),
                    'size': None,
                    'content_type': None,
                    'success': True,
                    'operation_type': "duplicate"
                }
            with self.lock:
                self.currently_processing.add(path)

            path = Path(path)

            content_type = detect_content_type(data)
            operation_type = 'write'

            # Check existence using cached directory contents
            if self.file_exists(path):
                return {
                    'status': 200,
                    'file': str(path),
                    'size': None,
                    'content_type': content_type,
                    'success': True,
                    'operation_type': "check"
                }
            else:
                # add to directory cache
                with self.lock:
                    self.directory_contents[str(path.parent)].add(str(path))
            
            result = None
            if content_type == 'url':
                operation_type = 'download'
                result = download_file(data, str(path))
                
            # elif content_type == 'application/octet-stream':
            #     operation_type = 'write_bytes'
            #     with open(path, 'wb') as f:
            #         f.write(data)
                
            else:# content_type == 'image/pil':
                data.save(path)
                
            # elif content_type == 'image/numpy':
            #     operation_type = 'save_numpy_image'
            #     Image.fromarray(
            #         data if data.dtype == np.uint8 else (data * 255).astype(np.uint8)
            #     ).save(path, format=force_format)
                
            # elif content_type == 'audio/numpy':
            #     operation_type = 'save_audio'
            #     sf.write(
            #         path, 
            #         data, 
            #         samplerate=kwargs.get('samplerate', 44100)
            #     )
                
            # elif content_type == 'tensor/pytorch':
            #     operation_type = 'save_tensor'
            #     torch.save(data, path)
                
            # else:
            #     raise ValueError(f"Unsupported data type: {content_type}")

            with self.lock:
                self.currently_processing.remove(path)

            if result is not None:
                return result
            return {
                'status': 200,
                'file': str(path),
                'size': None,
                'content_type': content_type,
                'success': True,
                'operation_type': operation_type
            }
            
        except Exception as e:
            return {
                'status': 500,
                'file': None,
                'error': str(e),
                'content_type': content_type,
                'success': False,
                'operation_type': operation_type
            }
        
    def _task_complete(self, future):
        # Ensure that futures are always cleaned up no matter what
        with self.lock:
            future_info = self.futures.pop(future, None)

        try:
            result = future.result()
        except Exception as e:
            # If the future.result() call itself raises an exception
            result = {
                'status': 500,
                'file': None,
                'error': str(e),
                'content_type': 'unknown',
                'success': False,
                'operation_type': 'unknown'
            }

        # At this point, we have a valid result dict or a synthetic one.
        with self.lock:
            self.completed += 1
            if result.get('success'):
                # Update EMA size if applicable
                if result.get('size') is not None:
                    size = result['size']
                    self.ema_size = (self.ema_alpha * size + 
                                     (1 - self.ema_alpha) * self.ema_size)
            else:
                self.failed += 1
                op_type = result.get('operation_type', 'unknown')
                self.failed_by_type[op_type] = self.failed_by_type.get(op_type, 0) + 1

                # Log the failed operation
                if future_info:
                    submitted_data = future_info['data']
                    source = submitted_data if isinstance(submitted_data, str) else "in-memory-object"
                    self.failed_operations.append(FailedOperation(
                        source=source,
                        target_path=future_info['path'],
                        operation_type=op_type,
                        id=future_info.get('id'),
                        error=result.get('error', 'Unknown error'),
                        timestamp=time.time()
                    ))
            
    def get_stats(self) -> OperationStats:
        return OperationStats(
            completed=self.completed,
            previously_completed=self.previous_completed,
            pending=len(self.futures),
            failed=self.failed,
            failed_by_type=dict(self.failed_by_type),
            success_rate=(self.completed - self.failed) / self.completed if self.completed else 0,
            avg_file_size=self.ema_size,
            elapsed_time=time.time() - self.start_time
        )

    def get_failed_operations(self, operation_type: Optional[str] = None) -> List[FailedOperation]:
        with self.lock:
            if operation_type is None:
                return list(self.failed_operations)
            return [op for op in self.failed_operations if op.operation_type == operation_type]

    def wait_for_completion(self, timeout=None) -> None:
        wait(self.futures, timeout=timeout)

    def shutdown(self, wait=True) -> None:
        self.executor.shutdown(wait=wait)

if __name__ == '__main__':
   dm = FileManager(num_workers=200)
   
   downloads = [
       {'url': 'https://via.placeholder.com/300.png/09f/fff', 'path': 'downloads/test.png'},
       {'url': 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4', 'path': 'downloads/test.mp4'}
   ]
   
   # Submit all downloads
   for download in downloads:
       dm.submit(download['url'], download['path'])
   
   # Optional: Monitor progress while downloads complete
   while True:
       stats = dm.get_stats()
       print(f"Completed: {stats.completed}, Pending: {stats.pending}, Success Rate: {stats.success_rate:.2%}")
       if stats.pending == 0:
           break
       time.sleep(1)
       
   # Get final results
   results = dm.wait_for_completion()
   dm.shutdown()