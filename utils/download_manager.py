from concurrent.futures import ThreadPoolExecutor, wait
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Optional, List, Deque
import requests
import time
import os
import shutil

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

class DownloadManager:
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
            
    def submit(self, url: str, path: str, id: Optional[str] = None) -> None:
        future = self.executor.submit(download_file, url, path)
        future.add_done_callback(self._task_complete)
        with self.lock:
            self.futures[future] = {'url': url, 'path': path, 'id': id}
            
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
                    self.failed_downloads.append(FailedDownload(
                        url=future_info['url'],
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
                    self.failed_downloads.append(FailedDownload(
                        url=future_info['url'],
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

    def shutdown(self):
        self.executor.shutdown(wait=True)

if __name__ == '__main__':
   dm = DownloadManager(num_workers=200)
   
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