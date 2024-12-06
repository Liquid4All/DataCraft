from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterator, Generator, Tuple
from collections import deque

@dataclass
class BaseData:
    """
    Integrated container for dataset loading, processing, and file management.
    Handles both the loading and batch processing of data, with built-in iteration
    and file management capabilities.
    """
    data_source: str
    data_type: str = "default"
    length: int = None
    default_batch_size: int = 10000

    def __init__(self, data_folder_path):
        # Initialize internal state
        self.data_folder_path = data_folder_path
        self._dataset = None
        self._current_index = 0
        self._files_to_save = deque()
        self._is_loaded = False
    
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
        return (self.batch_process(batch) for batch in base_iterator)
    
    def _iter_files(self) -> Generator[Tuple[str, str], None, None]:
        """
        Iterate through the files in the queue, removing them as we go.
        Returns a generator of (url, save_path) tuples.
        """
        while self._files_to_save:
            yield self._files_to_save.popleft()