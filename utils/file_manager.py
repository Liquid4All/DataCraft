from concurrent.futures import ThreadPoolExecutor, wait
from collections import deque
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
                'success': True
            }
            
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                return {
                    'status': getattr(e.response, 'status_code', 408),
                    'file': None,
                    'error': str(e),
                    'success': False
                }
            time.sleep(2 ** attempt)  # Exponential backoff

def detect_content_type(data: Any) -> str:
    """
    Detect the content type of the input data.
    Ordered by check speed (fastest first) and frequency of use.
    """
    # String (URL) check - fastest
    if isinstance(data, str):
        if data.lower().startswith(('http://', 'https://', 'ftp://')):
            return 'url'
        return 'text/plain'
    
    # Bytes check - very fast
    if isinstance(data, bytes):
        return 'application/octet-stream'
    
    # PIL Image - common in ML tasks
    if isinstance(data, Image.Image):
        return 'image/pil'
    
    # Torch Tensor - common in ML tasks
    if torch.is_tensor(data):
        return 'tensor/pytorch'
    
    # NumPy array - check last as it requires shape analysis
    if isinstance(data, np.ndarray):
        # Image: 2D or 3D array with appropriate shape
        if 2 <= data.ndim <= 3:
            if data.ndim == 2 or (data.ndim == 3 and data.shape[2] in [1, 3, 4]):
                return 'image/numpy'
        
        # Audio: 1D array or 2D array with 1-2 channels
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
    
    Args:
        data: The data to save
        path: Output file path
        force_format: Optional format to force (e.g., 'PNG' for images)
        **kwargs: Additional arguments passed to specific save functions
            - samplerate: for audio files (default: 44100)
            - quality: for JPEG images (0-100)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    content_type = detect_content_type(data)
    
    try:
        if content_type == 'url':
            return download_file(data, str(path))
            
        if content_type == 'application/octet-stream':
            with open(path, 'wb') as f:
                f.write(data)
            
        elif content_type == 'image/pil':
            save_kwargs = {}
            if force_format:
                save_kwargs['format'] = force_format
            if 'quality' in kwargs and path.suffix.lower() in ['.jpg', '.jpeg']:
                save_kwargs['quality'] = kwargs['quality']
            data.save(path, **save_kwargs)
            
        elif content_type == 'image/numpy':
            # Convert to PIL and save to handle various formats better
            Image.fromarray(
                data if data.dtype == np.uint8 else (data * 255).astype(np.uint8)
            ).save(path, format=force_format)
            
        elif content_type == 'audio/numpy':
            sf.write(
                path, 
                data, 
                samplerate=kwargs.get('samplerate', 44100)
            )
            
        elif content_type == 'tensor/pytorch':
            torch.save(data, path)
            
        else:  # unknown type - use pickle as fallback
            with open(path, 'wb') as f:
                pickle.dump(data, f, protocol=4)
        
        return {
            'status': 200,
            'file': str(path),
            'size': path.stat().st_size,
            'content_type': content_type,
            'success': True
        }
        
    except Exception as e:
        return {
            'status': 500,
            'file': None,
            'error': str(e),
            'content_type': content_type,
            'success': False
        }

@dataclass
class DownloadStats:
   completed: int
   pending: int
   failed: int
   success_rate: float
   avg_download_size: float
   elapsed_time: float

@dataclass
class FailedDownload:
    url: str
    path: str
    id: Optional[str]
    error: str

class FileManager:
    def __init__(self, num_workers=200, max_failed_history=1_000_000, ema_alpha=0.1):
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.futures = {}
        self.failed_downloads: Deque[FailedDownload] = deque(maxlen=max_failed_history)
        self.lock = Lock()
        self.start_time = time.time()
        
        # Running statistics
        self.completed = 0
        self.failed = 0
        self.ema_size = 0
        self.ema_alpha = ema_alpha
    
    def submit(self, data: Union[str, bytes, object], path: str, id: Optional[str] = None) -> None:
        # Instead of always downloading, we now handle all data via save_file
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
                    # Update exponential moving average of size
                    size = result.get('size', 0)
                    self.ema_size = (self.ema_alpha * size + 
                                     (1 - self.ema_alpha) * self.ema_size)
                else:
                    self.failed += 1
                    future_info = self.futures[future]
                    # We might not have a 'url' if data wasn't a URL; handle gracefully
                    submitted_data = future_info['data']
                    url = submitted_data if isinstance(submitted_data, str) else "in-memory-object"
                    self.failed_downloads.append(FailedDownload(
                        url=url,
                        path=future_info['path'],
                        id=future_info.get('id'),
                        error=result.get('error', 'Unknown error')
                    ))
                del self.futures[future]
        except Exception as e:
            with self.lock:
                self.completed += 1
                self.failed += 1
                if future in self.futures:
                    future_info = self.futures[future]
                    submitted_data = future_info['data']
                    url = submitted_data if isinstance(submitted_data, str) else "in-memory-object"
                    self.failed_downloads.append(FailedDownload(
                        url=url,
                        path=future_info['path'],
                        id=future_info.get('id'),
                        error=str(e)
                    ))
                    del self.futures[future]
            
    def get_stats(self) -> DownloadStats:
        with self.lock:
            return DownloadStats(
                completed=self.completed,
                pending=len(self.futures),
                failed=self.failed,
                success_rate=(self.completed - self.failed) / self.completed if self.completed else 0,
                avg_download_size=self.ema_size,
                elapsed_time=time.time() - self.start_time
            )

    def get_failed_downloads(self) -> List[FailedDownload]:
        with self.lock:
            return list(self.failed_downloads)

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