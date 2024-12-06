import pytest
import os
import tempfile
import time
from utils.download_manager import DownloadManager

# Adding some invalid URLs that should fail
TEST_FILES = {
    'images': [
        'https://picsum.photos/200/300',
        'https://placebear.com/g/200/200',
        'https://i.imgur.com/ex8f9sW.jpeg',
        'https://invalid.image.url/nonexistent.jpg'  # Should fail
    ],
    'videos': [
        'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
        'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4',
        'http://invalid.video.url/nonexistent.mp4'  # Should fail
    ],
    'pdfs': [
        'https://www.learningcontainer.com/wp-content/uploads/2019/09/sample-pdf-file.pdf',
        'https://www.learningcontainer.com/wp-content/uploads/2019/09/sample-pdf-download-10-mb.pdf',
        'https://invalid.pdf.url/nonexistent.pdf'  # Should fail
    ],
    'audio': [
        'https://file-examples.com/storage/fe3abb0cc967520c59b97f1/2017/11/file_example_MP3_700KB.mp3',
        'https://file-examples.com/storage/fe3abb0cc967520c59b97f1/2017/11/file_example_WAV_1MG.wav',
        'https://invalid.audio.url/nonexistent.mp3'  # Should fail
    ]
}

@pytest.fixture(scope="function")
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

@pytest.mark.parametrize("file_type", TEST_FILES.keys())
def test_bulk_download_with_tracking(temp_dir, file_type):
    manager = DownloadManager(num_workers=4, max_failed_history=10)
    
    downloads = [
        (url, os.path.join(temp_dir, f"{file_type}_{i}.{url.split('.')[-1]}"), f"{file_type}_{i}")
        for i, url in enumerate(TEST_FILES[file_type])
    ]
    
    for url, path, id in downloads:
        manager.submit(url=url, path=path, id=id)
    
    manager.wait_for_completion(timeout=60)
    failed_downloads = manager.get_failed_downloads()
    stats = manager.get_stats()
    
    assert stats.completed == len(downloads)
    assert stats.pending == 0
    assert len(failed_downloads) == 1  # Each category has one invalid URL
    
    manager.shutdown()

def test_download_all_types_concurrent(temp_dir):
    manager = DownloadManager(num_workers=8, max_failed_history=100)
    total_downloads = 0
    
    for file_type, urls in TEST_FILES.items():
        for i, url in enumerate(urls):
            path = os.path.join(temp_dir, f"{file_type}_{i}.{url.split('.')[-1]}")
            manager.submit(url=url, path=path, id=f"{file_type}_{i}")
            total_downloads += 1
    
    manager.wait_for_completion(timeout=120)
    stats = manager.get_stats()
    
    assert stats.completed == total_downloads
    assert stats.pending == 0
    assert stats.failed == len(TEST_FILES.keys())  # One failure per category
    
    manager.shutdown()

def test_failure_stack_limit():
    manager = DownloadManager(num_workers=2, max_failed_history=2)
    
    for i in range(5):
        manager.submit(
            url=f"https://invalid.url/{i}",
            path=f"/tmp/nonexistent_{i}",
            id=f"test_{i}"
        )
    
    manager.wait_for_completion(timeout=60)  # Allow failures to process
    failed_downloads = manager.get_failed_downloads()
    
    assert len(failed_downloads) == 2  # Limited by max_failed_history
    manager.shutdown()

def test_shutdown_wait(tmp_path):
    manager = DownloadManager(num_workers=1)
    manager.submit('https://invalid.url', str(tmp_path / 'test.txt'))
    
    start = time.time()
    manager.shutdown(wait=True)
    duration = time.time() - start
    
    assert duration >= 2  # Should wait for exponential backoff

def test_shutdown_no_wait(tmp_path):
    manager = DownloadManager(num_workers=1)
    manager.submit('https://invalid.url', str(tmp_path / 'test.txt'))
    
    start = time.time()
    manager.shutdown(wait=False)
    duration = time.time() - start
    
    assert duration < 1  # Should return immediately