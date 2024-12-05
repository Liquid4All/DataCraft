import pytest
import os
import json
import time
import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
import threading
from datetime import datetime
import random
import string

from data_manager import DataManager

@pytest.fixture
def db_path(tmp_path):
    """Fixture to provide a test database path in a temporary directory."""
    return str(tmp_path / "test_database.db")

@pytest.fixture
def data_manager(db_path):
    """Fixture to provide a DataManager instance."""
    manager = DataManager(db_path)
    yield manager
    manager.conn.close()

@pytest.fixture
def large_entries():
    """Fixture to provide a large number of diverse data entries."""
    entries = []
    for i in range(1000):  # Large enough to test batch processing
        entry = {
            'data_source': f'source{i % 5}',
            'group_id': f'group{i % 10}',
            'data_type': f'type{i % 3}',
            'data': json.dumps({
                'value': i,
                'timestamp': str(datetime.now()),
                'nested': {'field1': 'value1' * (i % 10)},  # Varying data sizes
                'array': list(range(i % 100))  # Varying array sizes
            }),
            'files': [f'file{j}.txt' for j in range(i % 5)]
        }
        entries.append(entry)
    return entries

class TestDataManagerInitialization:
    def test_init_with_invalid_path(self):
        """Test initialization with invalid database path."""
        with pytest.raises(sqlite3.OperationalError):
            DataManager("/nonexistent/path/db.sqlite")

    def test_init_with_corrupted_db(self, tmp_path):
        """Test initialization with corrupted database."""
        db_path = str(tmp_path / "corrupted.db")
        # Create corrupted database file
        with open(db_path, 'wb') as f:
            f.write(b'corrupted data')
        
        with pytest.raises(sqlite3.DatabaseError):
            DataManager(db_path)

    def test_concurrent_initialization(self, db_path):
        """Test concurrent initialization of multiple DataManager instances."""
        def create_manager():
            manager = DataManager(db_path)
            manager.conn.close()

        # Try to initialize multiple instances simultaneously
        threads = [threading.Thread(target=create_manager) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

class TestDataInsertion:
    def test_insert_empty_data(self, data_manager):
        """Test inserting empty data."""
        with pytest.raises(KeyError):
            data_manager.insert_data({})

    def test_insert_invalid_data_types(self, data_manager):
        """Test inserting data with invalid types."""
        invalid_entries = [
            {
                'data_source': None,  # Should be string
                'group_id': 123,      # Should be string
                'data_type': b'bytes',# Should be string
                'data': {},           # Should be JSON string
                'files': 'not_a_list' # Should be list
            }
        ]
        with pytest.raises(Exception):
            data_manager.insert_data(invalid_entries)

    def test_insert_extremely_large_data(self, data_manager):
        """Test inserting extremely large data entries."""
        large_data = {
            'data_source': 'source1',
            'group_id': 'group1',
            'data_type': 'type1',
            'data': json.dumps({'large': 'x' * 1000000}),  # 1MB of data
            'files': [f'file{i}.txt' for i in range(1000)]
        }
        uuids = data_manager.insert_data(large_data)
        assert len(uuids) == 1

class TestDataRetrieval:
    def test_get_nonexistent_data(self, data_manager):
        """Test retrieving data with non-existent filters."""
        data = data_manager.get_data(
            data_sources=['nonexistent'],
            group_ids=['nonexistent']
        )
        assert len(data) == 0

    def test_get_data_with_special_characters(self, data_manager):
        """Test retrieving data with special characters in filters."""
        special_entry = {
            'data_source': "source'with'quotes",
            'group_id': 'group;with;semicolons',
            'data_type': 'type--with--dashes',
            'data': json.dumps({'value': 1}),
            'files': ['file.txt']
        }
        data_manager.insert_data(special_entry)
        
        data = data_manager.get_data(
            data_sources=["source'with'quotes"],
            group_ids=['group;with;semicolons']
        )
        assert len(data) == 1

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, data_manager, large_entries):
        """Test concurrent read operations."""
        data_manager.insert_data(large_entries)

        async def concurrent_read():
            return data_manager.get_data(limit=10, offset=random.randint(0, 900))

        # Perform multiple concurrent reads
        tasks = [concurrent_read() for _ in range(20)]
        results = await asyncio.gather(*tasks)
        
        # Verify each result has correct number of entries
        assert all(len(result) <= 10 for result in results)

class TestIndexing:
    def test_index_with_duplicate_values(self, data_manager):
        """Test indexing with duplicate values."""
        entries = [
            {
                'data_source': 'source1',
                'group_id': 'group1',
                'data_type': 'type1',
                'data': json.dumps({'tag': 'duplicate'}),
                'files': ['file1.txt']
            },
            {
                'data_source': 'source2',
                'group_id': 'group2',
                'data_type': 'type2',
                'data': json.dumps({'tag': 'duplicate'}),
                'files': ['file2.txt']
            }
        ]
        data_manager.insert_data(entries)

        def tag_index(row):
            return [json.loads(row['data'])['tag']]

        data_manager.create_index('tag_index', tag_index)
        
        # Verify both entries are indexed under the same value
        rows = data_manager.get_data()
        for row in rows:
            values = data_manager.get_index_values_for_row('tag_index', row['uuid'])
            assert values == ['duplicate']

class TestDeletion:
    def test_delete_nonexistent_group(self, data_manager):
        """Test deleting non-existent group."""
        count = data_manager.delete_group('nonexistent')
        assert count == 0

    def test_delete_group_with_special_chars(self, data_manager):
        """Test deleting group with special characters."""
        special_entry = {
            'data_source': 'source1',
            'group_id': 'group;with;semicolons',
            'data_type': 'type1',
            'data': json.dumps({'value': 1}),
            'files': ['file.txt']
        }
        data_manager.insert_data(special_entry)
        count = data_manager.delete_group('group;with;semicolons')
        assert count == 1