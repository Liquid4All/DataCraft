import pytest
import os
import tempfile
import time
from utils.file_manager import FileManager

import numpy as np
import torch
from PIL import Image
import soundfile as sf
from pathlib import Path
import pickle
from typing import Union, Dict, Any
import tempfile
import shutil


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

# @pytest.mark.parametrize("file_type", TEST_FILES.keys())
# def test_bulk_download_with_tracking(temp_dir, file_type):
#     manager = FileManager(num_workers=4, max_failed_history=10)
    
#     downloads = [
#         (url, os.path.join(temp_dir, f"{file_type}_{i}.{url.split('.')[-1]}"), f"{file_type}_{i}")
#         for i, url in enumerate(TEST_FILES[file_type])
#     ]
    
#     for url, path, id in downloads:
#         manager.submit(url, path=path, id=id)
    
#     manager.wait_for_completion(timeout=60)
#     failed_downloads = manager.get_failed_operations()
#     stats = manager.get_stats()
    
#     assert stats.completed == len(downloads)
#     assert stats.pending == 0
#     assert len(failed_downloads) == 1  # Each category has one invalid URL
    
#     manager.shutdown()

# def test_download_all_types_concurrent(temp_dir):
#     manager = FileManager(num_workers=8, max_failed_history=100)
#     total_downloads = 0
    
#     for file_type, urls in TEST_FILES.items():
#         for i, url in enumerate(urls):
#             path = os.path.join(temp_dir, f"{file_type}_{i}.{url.split('.')[-1]}")
#             manager.submit(url, path=path, id=f"{file_type}_{i}")
#             total_downloads += 1
    
#     manager.wait_for_completion(timeout=120)
#     stats = manager.get_stats()
    
#     assert stats.completed == total_downloads
#     assert stats.pending == 0
#     assert stats.failed == len(TEST_FILES.keys())  # One failure per category
    
#     manager.shutdown()

# def test_failure_stack_limit():
#     manager = FileManager(num_workers=2, max_failed_history=2)
    
#     for i in range(5):
#         manager.submit(
#             f"https://invalid.url/{i}",
#             path=f"/tmp/nonexistent_{i}",
#             id=f"test_{i}"
#         )
    
#     manager.wait_for_completion(timeout=60)  # Allow failures to process
#     failed_downloads = manager.get_failed_operations()
    
#     assert len(failed_downloads) == 2  # Limited by max_failed_history
#     manager.shutdown()

# def test_shutdown_wait(tmp_path):
#     manager = FileManager(num_workers=1)
#     manager.submit('https://invalid.url', str(tmp_path / 'test.txt'))
    
#     start = time.time()
#     manager.shutdown(wait=True)
#     duration = time.time() - start
    
#     assert duration >= 2  # Should wait for exponential backoff

# def test_shutdown_no_wait(tmp_path):
#     manager = FileManager(num_workers=1)
#     manager.submit('https://invalid.url', str(tmp_path / 'test.txt'))
    
#     start = time.time()
#     manager.shutdown(wait=False)
#     duration = time.time() - start
    
#     assert duration < 1  # Should return immediately

def create_test_image_pil(size=(100, 100)):
    """Create a test PIL Image"""
    return Image.new('RGB', size, color='red')

def create_test_image_numpy(size=(100, 100)):
    """Create a test numpy array representing an image"""
    return np.ones((*size, 3), dtype=np.uint8) * 255

def create_test_audio():
    """Create a test audio signal"""
    duration = 1  # seconds
    samplerate = 44100
    t = np.linspace(0, duration, int(samplerate * duration))
    return np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

def create_test_tensor():
    """Create a test PyTorch tensor"""
    return torch.rand(3, 224, 224)

def test_submit_pil_image(temp_dir):
    """Test submitting PIL images"""
    img = create_test_image_pil()
    path = os.path.join(temp_dir, "test_pil.png")
    
    manager = FileManager(num_workers=4, max_failed_history=10)
    manager.submit(img, path=path, id="pil_test")
    manager.wait_for_completion(timeout=30)
    
    stats = manager.get_stats()
    failed = manager.get_failed_operations()
    
    assert stats.completed == 1
    assert stats.failed == 0
    assert len(failed) == 0
    assert os.path.exists(path)
    
    # Test JPEG with quality
    jpeg_path = os.path.join(temp_dir, "test_pil.jpg")
    manager.submit(img, path=jpeg_path, id="pil_jpeg_test")
    manager.wait_for_completion(timeout=30)
    assert os.path.exists(jpeg_path)

def test_submit_numpy_image(temp_dir):
    """Test submitting numpy array images"""
    img_array = create_test_image_numpy()
    path = os.path.join(temp_dir, "test_numpy.png")
    
    manager = FileManager(num_workers=4, max_failed_history=10)
    manager.submit(img_array, path=path, id="numpy_test")
    manager.wait_for_completion(timeout=30)
    
    stats = manager.get_stats()
    assert stats.completed == 1
    assert stats.failed == 0
    assert os.path.exists(path)
    
    # Test with forced format
    jpg_path = os.path.join(temp_dir, "test_numpy.jpg")
    manager.submit(img_array, path=jpg_path, id="numpy_jpg_test")
    manager.wait_for_completion(timeout=30)
    assert os.path.exists(jpg_path)

def test_submit_audio(temp_dir):
    """Test submitting audio data"""
    audio_signal = create_test_audio()
    path = os.path.join(temp_dir, "test_audio.wav")
    
    manager = FileManager(num_workers=4, max_failed_history=10)
    manager.submit(audio_signal, path=path, id="audio_test")
    manager.wait_for_completion(timeout=30)
    
    stats = manager.get_stats()
    assert stats.completed == 1
    assert stats.failed == 0
    assert os.path.exists(path)
    
    # Verify the saved audio file
    loaded_audio, sr = sf.read(path)
    assert sr == 44100
    # np.testing.assert_array_almost_equal(loaded_audio, audio_signal)

def test_submit_pytorch_tensor(temp_dir):
    """Test submitting PyTorch tensors"""
    tensor = create_test_tensor()
    path = os.path.join(temp_dir, "test_tensor.pt")
    
    manager = FileManager(num_workers=4, max_failed_history=10)
    manager.submit(tensor, path=path, id="tensor_test")
    manager.wait_for_completion(timeout=30)
    
    stats = manager.get_stats()
    assert stats.completed == 1
    assert stats.failed == 0
    assert os.path.exists(path)
    
    # Verify the saved tensor
    loaded_tensor = torch.load(path, weights_only=False)
    assert torch.all(loaded_tensor == tensor)

def test_submit_multiple_types_concurrent(temp_dir):
    """Test submitting multiple different types concurrently"""
    submissions = [
        (create_test_image_pil(), "test_pil.png", "pil_test"),
        (create_test_image_numpy(), "test_numpy.png", "numpy_test"),
        (create_test_audio(), "test_audio.wav", "audio_test"),
        (create_test_tensor(), "test_tensor.pt", "tensor_test")
    ]
    manager = FileManager(num_workers=4, max_failed_history=10)
    for data, filename, id in submissions:
        path = os.path.join(temp_dir, filename)
        manager.submit(data, path=path, id=id)
    
    manager.wait_for_completion(timeout=60)
    
    stats = manager.get_stats()
    assert stats.completed == len(submissions)
    assert stats.failed == 0
    assert stats.pending == 0
    
    # Verify all files exist
    for _, filename, _ in submissions:
        assert os.path.exists(os.path.join(temp_dir, filename))

def test_submit_error_handling(temp_dir):
    """Test error handling for invalid submissions"""
    # Test with invalid path
    img = create_test_image_pil()
    invalid_path = "/invalid -3;a'df;lk3/path/test.png"

    manager = FileManager(num_workers=4, max_failed_history=10)
    manager.submit(img, path=invalid_path, id="invalid_path_test")
    manager.wait_for_completion(timeout=30)
    
    stats = manager.get_stats()
    failed = manager.get_failed_operations()
    assert stats.failed == 1
    assert len(failed) == 1
    
    # Test with invalid data type
    class UnserializableObject:
        def __getstate__(self):
            raise TypeError("Cannot serialize")
    
    path = os.path.join(temp_dir, "invalid_object.pkl")
    manager.submit(UnserializableObject(), path=path, id="invalid_object_test")
    manager.wait_for_completion(timeout=30)
    
    stats = manager.get_stats()
    failed = manager.get_failed_operations()
    assert stats.failed == 2  # Including previous failure
    assert len(failed) == 2

def test_submit_large_batch(temp_dir):
    """Test submitting a large batch of files"""
    num_files = 20
    submissions = []
    
    for i in range(num_files):
        img = create_test_image_pil((10, 10))  # Small size for quick processing
        path = os.path.join(temp_dir, f"batch_test_{i}.png")
        submissions.append((img, path, f"batch_{i}"))
    
    manager = FileManager(num_workers=4, max_failed_history=10)
    for data, path, id in submissions:
        manager.submit(data, path=path, id=id)
    
    manager.wait_for_completion(timeout=120)
    
    stats = manager.get_stats()
    assert stats.completed == num_files
    assert stats.failed == 0
    assert stats.pending == 0
    
    # Verify all files exist
    for _, path, _ in submissions:
        assert os.path.exists(path)