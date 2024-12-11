from concurrent.futures import ThreadPoolExecutor, wait
from collections import deque, OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Optional, List, Deque, Dict, Union, Any
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
    pending: int
    failed: int
    failed_by_type: Dict[str, int]  # Tracks failures by operation type
    success_rate: float
    avg_file_size: float
    elapsed_time: float

def download_file(url, path, headers=None, max_retries=3, timeout=30):
    default_headers = {
        'User-Agent': 'Googlebot-Image/1.0',
        'X-Forwarded-For': '64.18.15.200'
    }
    headers = headers or default_headers
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=timeout)
            response.raise_for_status()
            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            response.raw.decode_content = True
            with open(path, 'wb') as f:
                shutil.copyfileobj(response.raw, f, length=65536)
                    
            return {
                'status': response.status_code,
                'file': path,
                'size': os.path.getsize(path),
                'content_type': response.headers.get('content-type'),
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

def save_file(
    data: Union[str, bytes, object],
    path: str,
    force_format: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Save data to a file with support for core ML libraries.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    content_type = detect_content_type(data)
    operation_type = 'write'

    # if file already exists, return success
    if path.exists():
        return {
            'status': 200,
            'file': str(path),
            'size': path.stat().st_size,
            'content_type': content_type,
            'success': True,
            'operation_type': "check"
        }
    
    try:
        if content_type == 'url':
            operation_type = 'download'
            return download_file(data, str(path))
            
        if content_type == 'application/octet-stream':
            operation_type = 'write_bytes'
            with open(path, 'wb') as f:
                f.write(data)
            
        elif content_type == 'image/pil':
            operation_type = 'save_image'
            save_kwargs = {}
            if force_format:
                save_kwargs['format'] = force_format
            if 'quality' in kwargs and path.suffix.lower() in ['.jpg', '.jpeg']:
                save_kwargs['quality'] = kwargs['quality']
            data.save(path, **save_kwargs)
            
        elif content_type == 'image/numpy':
            operation_type = 'save_numpy_image'
            Image.fromarray(
                data if data.dtype == np.uint8 else (data * 255).astype(np.uint8)
            ).save(path, format=force_format)
            
        elif content_type == 'audio/numpy':
            operation_type = 'save_audio'
            sf.write(
                path, 
                data, 
                samplerate=kwargs.get('samplerate', 44100)
            )
            
        elif content_type == 'tensor/pytorch':
            operation_type = 'save_tensor'
            torch.save(data, path)
            
        else:
            operation_type = 'save_pickle'
            with open(path, 'wb') as f:
                pickle.dump(data, f, protocol=4)
        
        return {
            'status': 200,
            'file': str(path),
            'size': path.stat().st_size,
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

class FileManager:
    def __init__(self, num_workers=200, max_failed_history=1_000_000, ema_alpha=0.1):
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.futures = {}
        self.failed_operations: Deque[FailedOperation] = deque(maxlen=max_failed_history)
        self.lock = Lock()
        self.start_time = time.time()
        
        # Cache for recent requests
        self.request_cache = OrderedDict()
        self.max_cache_size = num_workers  # Cache size matches worker count
        
        # Running statistics
        self.completed = 0
        self.failed = 0
        self.failed_by_type = {}  # Track failures by operation type
        self.ema_size = 0
        self.ema_alpha = ema_alpha
    
    def submit(self, data: Union[str, bytes, object], path: str, id: Optional[str] = None) -> None:
        with self.lock:
            # Check if this request is already in progress
            if path in self.request_cache:
                self.cache_hits += 1
                # Move the existing future to the end (most recent)
                self.request_cache.move_to_end(path)
                return
            
            self.cache_misses += 1
            
            # Add placeholder to cache before submitting
            self.request_cache[path] = None
            
            # If cache is too large, remove oldest entry
            if len(self.request_cache) > self.max_cache_size:
                self.request_cache.popitem(last=False)
            
            try:
                # Submit the task
                future = self.executor.submit(save_file, data, path)
                future.add_done_callback(self._task_complete)
                
                # Update cache and futures with the actual future
                self.request_cache[path] = future
                self.futures[future] = {'data': data, 'path': path, 'id': id}
            except Exception as e:
                # If submission fails, remove from cache
                self.request_cache.pop(path, None)
                raise e

    def submit(self, data: Union[str, bytes, object], path: str, id: Optional[str] = None) -> None:
        future = self.executor.submit(save_file, data, path)
        future.add_done_callback(self._task_complete)
        with self.lock:
            self.futures[future] = {'data': data, 'path': path, 'id': id}
            
    def _task_complete(self, future):
        try:
            result = future.result()
            with self.lock:
                self.completed += 1
                if result['success']:
                    size = result.get('size', 0)
                    self.ema_size = (self.ema_alpha * size + 
                                   (1 - self.ema_alpha) * self.ema_size)
                else:
                    self.failed += 1
                    operation_type = result.get('operation_type', 'unknown')
                    self.failed_by_type[operation_type] = self.failed_by_type.get(operation_type, 0) + 1
                    
                    future_info = self.futures[future]
                    submitted_data = future_info['data']
                    source = submitted_data if isinstance(submitted_data, str) else "in-memory-object"
                    
                    self.failed_operations.append(FailedOperation(
                        source=source,
                        target_path=future_info['path'],
                        operation_type=operation_type,
                        id=future_info.get('id'),
                        error=result.get('error', 'Unknown error'),
                        timestamp=time.time()
                    ))
                del self.futures[future]
        except Exception as e:
            with self.lock:
                self.completed += 1
                self.failed += 1
                operation_type = 'unknown'
                self.failed_by_type[operation_type] = self.failed_by_type.get(operation_type, 0) + 1
                
                if future in self.futures:
                    future_info = self.futures[future]
                    submitted_data = future_info['data']
                    source = submitted_data if isinstance(submitted_data, str) else "in-memory-object"
                    
                    self.failed_operations.append(FailedOperation(
                        source=source,
                        target_path=future_info['path'],
                        operation_type=operation_type,
                        id=future_info.get('id'),
                        error=str(e),
                        timestamp=time.time()
                    ))
                    del self.futures[future]
            
    def get_stats(self) -> OperationStats:
        with self.lock:
            return OperationStats(
                completed=self.completed,
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