from dataclasses import dataclass, field
from typing import List, Dict, Iterator, Generator, Tuple, Any
from collections import deque
import pyarrow as pa

class HFStaticDataIterator:
    """
    Iterator that uses select() for efficient batch processing of huggingface Datasets.
    Returns batches in standard Python types rather than PyArrow format.
    
    Args:
        dataset: A HuggingFace dataset object
        
    Returns:
        Batches of data in standard Python types (lists, dicts, etc.)
    """
    def __init__(self, dataset):
        self.dataset = dataset
    
    def _convert_to_python_type(self, value):
        """Helper method to convert various types to Python native types."""
        if isinstance(value, (pa.Array, pa.ChunkedArray)):
            return value.to_pandas().tolist()
        elif isinstance(value, list):
            return value
        else:
            return [value]  # Single value case
    
    def iter(self, batch_size):
        current_index = 0
        dataset_size = len(self.dataset)
        
        while current_index < dataset_size:
            end_idx = min(current_index + batch_size, dataset_size)
            batch = self.dataset.select(range(current_index, end_idx))
            
            # Convert each feature to Python types
            batch_dict = {
                key: self._convert_to_python_type(batch[key])
                for key in batch.features.keys()
            }
            
            yield batch_dict
            current_index = end_idx
    
@dataclass
class BaseData:
    """
    Integrated container for dataset loading, processing, and file management.
    Handles both the loading and batch processing of data, with built-in iteration
    and file management capabilities.
    """
    # Required fields (often set by superclass)
    data_source: str
    data_type: str = "default"
    length: int = None
    default_batch_size: int = 100000

    # Directly managed by data_manager
    data_folder_path: str = field(default=None)
    dataset_subset: Any = field(default=None)

    # Objects managed by the class
    _dataset = None
    _current_index = 0
    _files_to_save = deque()
    _is_loaded = False

    def __init__(self):
        pass
    
    def load(self) -> None:
        """
        Load the dataset. Must be overridden for custom loading logic.

        Returns:
            Dataset object (must have a '.iter(batch_size)' method)
        """
        raise NotImplementedError
    
    def batch_process(self, batch: Dict[str, List]) -> Dict[str, List]:
        """
        Process a batch of data. Must be overridden for custom processing logic.
        
        Returns:
            Dictionary with processed data
                Processed data: Dict[str, List]
        """
        raise NotImplementedError

    def add_file(self, url: str, save_path: str) -> None:
        """
        Add a file to the save queue.
        """
        self._files_to_save.append((url, save_path))
    
    def info(self) -> None:
        """
        Print information about the current state.
        """
        print(f"Data Source: {self.data_source}")
        print(f"Data Type: {self.data_type}")
        print(f"Dataset loaded: {self._is_loaded}")
        if self._is_loaded:
            print(f"\tTotal samples: {len(self._dataset)}")
        print(f"Number Files queued for saving: {len(self._files_to_save)}")
    
    def _load(self) -> None:
        """
        Internal method to load the dataset and set internal state.
        """
        self._dataset = self.load()
        self._is_loaded = True
        self._current_index = 0

    def _iter(self, batch_size: int = None) -> Iterator[Dict[str, List]]:
        """
        Create an iterator with specified batch size.
        Automatically loads data if not already loaded.
        """
        if batch_size is None:
            batch_size = self.default_batch_size

        if not self._is_loaded:
            self._load()

        base_iterator = self._dataset.iter(batch_size=batch_size)
        future_iterator = (self.batch_process(batch) for batch in base_iterator)
        return future_iterator
    
    def _iter_files(self) -> Generator[Tuple[str, str], None, None]:
        """
        Iterate through the files in the queue, removing them as we go.
        Returns a generator of (url, save_path) tuples.
        """
        while self._files_to_save:
            yield self._files_to_save.popleft()
    
    def __setattr__(self, name, value):
        self.__dict__[name] = value