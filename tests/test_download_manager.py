import pytest
import os
import tempfile
from datetime import datetime
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
    
    results = manager.wait_for_completion(timeout=60)
    failed_downloads = manager.get_failed_downloads()
    stats = manager.get_stats()
    
    # Basic completion checks
    assert stats.completed == len(downloads)
    assert stats.pending == 0
    
    # Check failed downloads tracking
    assert len(failed_downloads) == 1  # Each category has one invalid URL
    assert all(hasattr(fd, 'id') for fd in failed_downloads)
    assert all(hasattr(fd, 'error') for fd in failed_downloads)
    
    # Check successful downloads
    successful_files = [r['file'] for r in results if r.get('success', False)]
    assert len(successful_files) == len(downloads) - 1
    for file_path in successful_files:
        assert os.path.exists(file_path)
        assert os.path.getsize(file_path) > 0

    manager.shutdown()

def test_download_all_types_with_failure_tracking(temp_dir):
    manager = DownloadManager(num_workers=8, max_failed_history=100)
    total_downloads = 0
    
    for file_type, urls in TEST_FILES.items():
        for i, url in enumerate(urls):
            id = f"{file_type}_{i}_{datetime.now().timestamp()}"
            path = os.path.join(temp_dir, f"{file_type}_{i}.{url.split('.')[-1]}")
            manager.submit(url=url, path=path, id=id)
            total_downloads += 1
    
    results = manager.wait_for_completion(timeout=120)
    failed_downloads = manager.get_failed_downloads()
    stats = manager.get_stats()
    
    # Verify stats
    assert stats.completed == total_downloads
    assert stats.pending == 0
    assert stats.failed == len(TEST_FILES.keys())  # One failure per category
    
    # Verify failed downloads tracking
    assert len(failed_downloads) == len(TEST_FILES.keys())
    assert all(fd.error for fd in failed_downloads)
    
    # Verify successful downloads
    successful = [r for r in results if r.get('success', False)]
    assert len(successful) == total_downloads - len(TEST_FILES.keys())
    
    for result in successful:
        assert os.path.exists(result['file'])
        assert os.path.getsize(result['file']) > 0
    
    manager.shutdown()

def test_failure_stack_limit():
    manager = DownloadManager(num_workers=2, max_failed_history=2)
    
    # Submit more failed downloads than the limit
    for i in range(5):
        manager.submit(
            url=f"https://invalid.url/{i}",
            path=f"/tmp/nonexistent_{i}",
            id=f"test_{i}"
        )
    
    manager.wait_for_completion(timeout=10)
    failed_downloads = manager.get_failed_downloads()
    
    assert len(failed_downloads) == 2  # Should be limited by max_failed_history
    manager.shutdown()

@pytest.fixture(scope="function", autouse=True)
def cleanup():
    yield
    import psutil
    current = psutil.Process()
    for child in current.children(recursive=True):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass