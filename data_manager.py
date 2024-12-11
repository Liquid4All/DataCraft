import asyncio
from datetime import datetime
import json
import os
import importlib.util
import inspect
import sqlite3
import threading
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from tqdm import tqdm
import hashlib
import uuid
from utils.file_manager import FileManager
from utils.format import sanitize_filename

class SQLBaseManager:
    """
    Base class containing internal methods for managing database connections,
    table creation, and index processing.
    """
    BASE_TABLE_COLUMNS = [('uuid', 'TEXT PRIMARY KEY'),
                            ('data_source', 'TEXT'),
                            ('group_id', 'TEXT'),
                            ('data_type', 'TEXT'),
                            ('data', 'TEXT'),
                            ('files', 'TEXT')]

    def _connect_db(self):
        """Establish connection to the SQLite database."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        # Commit the settings
        self.conn.commit()

    def _create_main_table(self):
        """Create the main data table and required indexes if they don't exist."""
        create_table_sql = "CREATE TABLE IF NOT EXISTS main_table ("
        create_table_sql += ", ".join([f'{col[0]} {col[1]}' for col in self.BASE_TABLE_COLUMNS])
        create_table_sql += ");"
        self.cursor.execute(create_table_sql)

        # Index on data_source and group_id
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_data_source ON main_table (data_source);')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_group_id ON main_table (group_id);')
        self.conn.commit()
    
    def _insert_data(self, entries: Dict[str, List[Any]], batch_size: int = 100000) -> Tuple[bool, List[str]]:
        """
        Insert batch of data entries into database, skipping existing records.
        
        Args:
            entries (Dict[str, List[Any]]): Dictionary containing lists of data to insert
            batch_size (int): Number of records to insert in each batch
                
        Returns:
            Tuple[bool, List[str]]: (Success status, List of skipped UUIDs)
        """
        insert_sql = """
        INSERT OR IGNORE INTO main_table (uuid, data_source, group_id, data_type, data, files)
        VALUES (?, ?, ?, ?, ?, ?);
        """
        
        # Convert entries to list of tuples for batched processing
        data_list = list(zip(
            entries["uuid"],
            entries["data_source"],
            entries["group_id"],
            entries["data_type"],
            entries["data"],
            entries["files"]
        ))
        
        total_records = len(data_list)
        records_inserted = 0
        skipped_uuids = []
        
        # First, get existing UUIDs
        existing_uuids = set()
        self.cursor.execute("SELECT uuid FROM main_table WHERE uuid IN ({})".format(
            ','.join('?' * len(entries["uuid"]))
        ), entries["uuid"])
        existing_uuids.update(row[0] for row in self.cursor.fetchall())
        
        with self.lock:
            while records_inserted < total_records:
                # Get the next batch
                batch = data_list[records_inserted:records_inserted + batch_size]
                
                # Track which UUIDs in this batch already exist
                batch_uuids = [item[0] for item in batch]
                skipped_uuids.extend([uuid for uuid in batch_uuids if uuid in existing_uuids])
                
                # Process the batch within a transaction
                self.conn.execute('BEGIN IMMEDIATE TRANSACTION;')
                try:
                    self.cursor.executemany(insert_sql, batch)
                    self.conn.commit()
                    records_inserted += len(batch)
                except Exception as e:
                    self.conn.rollback()
                    raise e
        
        if skipped_uuids:
            print(f"Skipped {len(skipped_uuids)} existing records.")
            
        return True, skipped_uuids

    def _create_index_tables(self, index_name: str):
        """Create the forward and backward index tables."""
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS index_forward_{index_name} (
                row_id TEXT,
                index_value TEXT,
                FOREIGN KEY(row_id) REFERENCES main_table(uuid)
            );
        ''')
        
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS index_backward_{index_name} (
                index_value TEXT,
                row_id TEXT,
                FOREIGN KEY(row_id) REFERENCES main_table(uuid)
            );
        ''')
        
        self.cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_forward_{index_name} ON index_forward_{index_name} (row_id);')
        self.cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_backward_{index_name} ON index_backward_{index_name} (index_value);')
        self.conn.commit()

    async def _process_index_batch(self, index_name: str, batch_uuids: List[str], 
                             index_function: Callable) -> List[tuple]:
        """
        Process a batch of rows for indexing, supporting both synchronous and asynchronous functions.
        Synchronous functions are processed sequentially, while async functions are processed in parallel.

        Args:
            index_name (str): Name of the index
            batch_uuids (list): List of UUIDs to process
            index_function (callable): Function to generate index values, can be sync or async

        Returns:
            list: List of (uuid, index_value) tuples
        """
        # Fetch all rows for the batch
        placeholders = ','.join(['?' for _ in batch_uuids])
        query = f'SELECT * FROM main_table WHERE uuid IN ({placeholders})'
        self.cursor.execute(query, batch_uuids)
        rows = self.cursor.fetchall()

        # Convert rows to dictionaries
        row_dicts = [{
            'uuid': row[0],
            'data_source': row[1],
            'group_id': row[2],
            'data_type': row[3],
            'data': row[4],
            'files': row[5].split(',')
        } for row in rows]

        is_async = inspect.iscoroutinefunction(index_function)
        results = []

        if is_async:
            # For async functions, process all rows in parallel
            async def process_row_async(row_dict):
                try:
                    index_values = await index_function(row_dict)
                    return row_dict['uuid'], index_values
                except Exception:
                    return row_dict['uuid'], None

            tasks = [process_row_async(row_dict) for row_dict in row_dicts]
            results = await asyncio.gather(*tasks)
        else:
            # For sync functions, process rows sequentially
            def process_row_sync(row_dict):
                try:
                    index_values = index_function(row_dict)
                    return row_dict['uuid'], index_values
                except Exception:
                    return row_dict['uuid'], None

            # Process each row sequentially
            results = [process_row_sync(row_dict) for row_dict in row_dicts]

        return results

####################################################################################################

class _ErrorFileManager:
    def __getattr__(self, _):
        raise ValueError("DataManager Argument 'download_manager_workers' must be greater than 0 to use download manager features")
    
class DataManager(SQLBaseManager):
    """
    A class for managing data storage and retrieval using SQLite with support for
    batch operations, custom indexing, and async processing.
    
    Attributes:
        db_path (str): Path to the SQLite database file
        conn (sqlite3.Connection): Database connection object
        cursor (sqlite3.Cursor): Database cursor object
    """

    def __init__(self, db_path: str, download_manager_workers: int = 0):
        """
        Initialize the DataManager with a database path.

        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self.lock = threading.Lock()
        self._connect_db()
        self._create_main_table()
        if download_manager_workers > 0:
            self.file_manager = FileManager(num_workers=download_manager_workers)
        else:
            self.file_manager = _ErrorFileManager()

    def insert_data(self, entries: Dict[str, List[Any]], data_source: Optional[str] = None, 
                group_id: Optional[str] = None, data_type: Optional[str] = None):
        """Insert batch of data entries into database with deterministic UUIDs."""
        num_entries = len(entries['data'])

        if "data" not in entries:
            raise ValueError("Data must be provided in entries.")
        if "data_source" not in entries:
            if not data_source:
                raise ValueError("data_source must be provided if not included in entries.")
            entries["data_source"] = [data_source] * num_entries
        if "group_id" not in entries:
            if not group_id:
                raise ValueError("group_id must be provided if not included in entries.")
            entries["group_id"] = [group_id] * num_entries
        if "data_type" not in entries:
            if not data_type:
                raise ValueError("data_type must be provided if not included in entries.")
            entries["data_type"] = [data_type] * num_entries
        if "files" not in entries:
            entries["files"] = [json.dumps([])] * num_entries

        if "uuid" not in entries:
            # Generate deterministic UUIDs based on data, data_source, and files
            entries["uuid"] = [
                str(uuid.UUID(hex=hashlib.md5( # Use MD5 hash rather than sha256 for speed
                    f"{str(entries['data'][i])}{entries['data_source'][i]}{entries['files'][i]}".encode()
                ).hexdigest()))
                for i in range(num_entries)
            ]
        return self._insert_data(entries)
    
    def process_dataset(
        self,
        processor_file: str,
        data_folder_path: str,
        dataset_subset: Optional[str] = None,
        group_id: Optional[str] = None,
        batch_size: Optional[int] = None,
        sample: bool = False,
        skip_file_download: bool = False
    ) -> None:
        """
        Processes a dataset using a DataInfo class implementation.
        
        Args:
            processor_file: Path to Python file containing Data implementation
            data_folder_path: Path to the folder where data files will be saved
            dataset_subset: Name of the dataset subset to process
            group_id: Group ID for data insertion (defaults to data-name-YYYYMMDD)
            batch_size: Forces a given batch size for processing
            sample: If True, only processes the first batch of data
            skip_file_download: If True, skips downloading files
        """
        # Make sure the file exists
        if not os.path.exists(processor_file):
            raise FileNotFoundError(f"Processor file not found: {processor_file}")
        
        # Import the DataInfo class from the processor file
        spec = importlib.util.spec_from_file_location("processor", processor_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the DataInfo class and set data_folder_path and dataset_subset BEFORE Initialization so they can be used in any custom init function
        DataClass = getattr(module, "Data")
        DataClass.data_folder_path = data_folder_path
        DataClass.dataset_subset = dataset_subset
        data_info = DataClass()

        # replace batch_size with default_batch_size if not provided
        if batch_size is None:
            batch_size = data_info.default_batch_size

        # if group_id is None, then set it to data_source-YYYYMMDD
        if group_id is None:
            group_id = f"{sanitize_filename(data_info.data_source)}-{datetime.now().strftime('%Y%m%d')}"

        # Process the dataset with progress bar
        with tqdm(total=data_info.length, desc="Processing Rows", unit="rows") as pbar:
            for batch in data_info._iter(batch_size):
                self.insert_data(
                    entries=batch,
                    data_source=data_info.data_source,
                    group_id=group_id,
                    data_type=data_info.data_type
                )
                
                # Save any files that were queued
                if not skip_file_download:
                    for file_data, path in data_info._iter_files():
                        self.file_manager.submit(file_data, path)
                
                # Update progress bar
                pbar.update(len(batch["uuid"]))

                if sample:
                    break

        # Wait for all downloads to complete
        self.file_manager.wait_for_completion()
                
    def create_index(self, index_name: str, index_function: Callable,
                data_sources: Optional[List[str]] = None,
                group_ids: Optional[List[str]] = None,
                data_types: Optional[List[str]] = None,
                batch_size: Optional[int] = None):
        """
        Create a custom index using the provided indexing function.
        
        Args:
            index_name (str): Name of the index to create
            index_function (callable): Function that takes a row dict and returns a list of index values
            data_sources (list, optional): Filter by data sources
            group_ids (list, optional): Filter by group IDs
            data_types (list, optional): Filter by data types
            batch_size (int, optional): Number of rows to process in each batch
        """
        with self.lock:
            # Create index tables if they don't exist
            self._create_index_tables(index_name)
            
            # Build query to get unprocessed rows
            query_parts = ['SELECT uuid FROM main_table']
            where_conditions = []
            params = []
            
            # Add NOT EXISTS condition to exclude already processed rows
            where_conditions.append(f'''
                NOT EXISTS (
                    SELECT 1 FROM index_forward_{index_name}
                    WHERE row_id = main_table.uuid
                )
            ''')
            
            if data_sources:
                placeholders = ','.join(['?' for _ in data_sources])
                where_conditions.append(f'data_source IN ({placeholders})')
                params.extend(data_sources)
                
            if group_ids:
                placeholders = ','.join(['?' for _ in group_ids])
                where_conditions.append(f'group_id IN ({placeholders})')
                params.extend(group_ids)
                
            if data_types:
                placeholders = ','.join(['?' for _ in data_types])
                where_conditions.append(f'data_type IN ({placeholders})')
                params.extend(data_types)
                
            query = query_parts[0]
            if where_conditions:
                query += ' WHERE ' + ' AND '.join(where_conditions)
                
            # Get all relevant UUIDs
            self.cursor.execute(query, params)
            all_uuids = [row[0] for row in self.cursor.fetchall()]
            
            if not all_uuids:
                return
                
            # Process in batches
            batch_size = batch_size or len(all_uuids)
            
            for i in range(0, len(all_uuids), batch_size):
                batch_uuids = all_uuids[i:i + batch_size]
                
                # Initialize batch in forward index with empty lists
                self.conn.execute('BEGIN TRANSACTION;')
                self.cursor.executemany(
                    f'INSERT INTO index_forward_{index_name} (row_id, index_value) VALUES (?, ?)',
                    [(uuid_str, '') for uuid_str in batch_uuids]
                )
                self.conn.commit()
                
                # Process batch
                results = asyncio.run(self._process_index_batch(index_name, batch_uuids, index_function))
                
                # Update indexes with results
                forward_data = []
                backward_data = []
                
                for uuid_str, index_values in results:
                    if index_values is None:  # Failed processing
                        forward_data.append((uuid_str, None))
                    elif index_values:  # Successful processing with values
                        for value in index_values:
                            forward_data.append((uuid_str, value))
                            backward_data.append((value, uuid_str))
                            
                # Update the indexes
                self.conn.execute('BEGIN TRANSACTION;')
                try:
                    # Clear temporary entries
                    self.cursor.execute(
                        f'DELETE FROM index_forward_{index_name} WHERE row_id IN ({",".join("?" * len(batch_uuids))})',
                        batch_uuids
                    )
                    
                    # Insert processed results
                    if forward_data:
                        self.cursor.executemany(
                            f'INSERT INTO index_forward_{index_name} (row_id, index_value) VALUES (?, ?)',
                            forward_data
                        )
                    if backward_data:
                        self.cursor.executemany(
                            f'INSERT INTO index_backward_{index_name} (index_value, row_id) VALUES (?, ?)',
                            backward_data
                        )
                    self.conn.commit()
                except Exception as e:
                    self.conn.rollback()
                    raise e

    def delete_index(self, index_name):
        """
        Delete a index and its associated tables. Warns if tables don't exist.
        
        Args:
            index_name (str): Name of the index to delete
            
        Returns:
            bool: True if tables were deleted, False if no tables were found
        """
        with self.lock:
            # Check if tables exist before trying to drop them
            self.cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' 
                AND (name=? OR name=?);
            """, (f'index_forward_{index_name}', f'index_backward_{index_name}'))
            
            existing_tables = self.cursor.fetchall()
            
            if not existing_tables:
                return False
                
            self.cursor.execute(f'DROP TABLE IF EXISTS index_forward_{index_name};')
            self.cursor.execute(f'DROP TABLE IF EXISTS index_backward_{index_name};')
            self.conn.commit()
            return True

    def get_data(self, data_sources=None, group_ids=None, limit=0, offset=None):
        """
        Fetch data with optional filtering by data sources and/or group IDs, with pagination support.
        
        Args:
            data_sources (list): Optional list of data sources to filter by
            group_ids (list): Optional list of group IDs to filter by
            limit (int): Maximum number of records to return (0 for no limit)
            offset (int): Number of records to skip (None for no offset)
        
        Returns:
            list: List of dictionaries containing query results, with column names as keys
        """
        query_parts = ['SELECT * FROM main_table']
        where_conditions = []
        params = []
        
        # Add data source filter if provided
        if data_sources:
            placeholders = ','.join(['?'] * len(data_sources))
            where_conditions.append(f'data_source IN ({placeholders})')
            params.extend(data_sources)
        
        # Add group ID filter if provided
        if group_ids:
            placeholders = ','.join(['?'] * len(group_ids))
            where_conditions.append(f'group_id IN ({placeholders})')
            params.extend(group_ids)
        
        # Combine WHERE conditions if any exist
        if where_conditions:
            query_parts.append('WHERE ' + ' AND '.join(where_conditions))
        
        # Add LIMIT clause if specified
        if limit > 0:
            query_parts.append('LIMIT ?')
            params.append(limit)
        
        # Add OFFSET clause if specified
        if offset is not None:
            query_parts.append('OFFSET ?')
            params.append(offset)
        
        # Combine all parts and execute query
        query_sql = ' '.join(query_parts)
        self.cursor.execute(query_sql, params)
        
        # Get column names from BASE_TABLE_COLUMNS
        columns = [col[0] for col in self.BASE_TABLE_COLUMNS]
        
        # Convert query results to list of dictionaries
        results = []
        for row in self.cursor.fetchall():
            row_dict = {columns[i]: value for i, value in enumerate(row)}
            results.append(row_dict)
        
        return results
    
    def sample_data(self, data_sources=None, group_ids=None, limit=10, offset=None):
        """
        Fetch a sample of records for each combination of data source and group ID,
        or for each data source/group ID individually if only one filter is provided.
        
        Args:
            limit (int): Number of records to fetch per combination
            data_sources (list): Optional list of data sources to sample from
            group_ids (list): Optional list of group IDs to sample from
        
        Returns:
            list: List of dictionaries containing sample data, with column names as keys
        """
        results = []
        
        if data_sources and group_ids:
            # Get samples for each combination of data source and group ID
            for data_source in data_sources:
                for group_id in group_ids:
                    sample = self.get_data(
                        data_sources=[data_source],
                        group_ids=[group_id],
                        limit=limit,
                        offset=offset
                    )
                    results.extend(sample)
                    
        elif data_sources:
            # Get samples for each data source
            for data_source in data_sources:
                sample = self.get_data(
                    data_sources=[data_source],
                    limit=limit,
                    offset=offset
                )
                results.extend(sample)
                
        elif group_ids:
            # Get samples for each group ID
            for group_id in group_ids:
                sample = self.get_data(
                    group_ids=[group_id],
                    limit=limit,
                    offset=offset
                )
                results.extend(sample)
                
        else:
            # If no filters provided, get overall sample
            results = self.get_data(limit=limit)
        
        return results

    def delete(self, group_ids: Union[str, List[str]] = None, data_sources: Union[str, List[str]] = None, batch_size: int = 1000) -> int:
        """
        Delete all data entries associated with specified group IDs and/or data sources, including their index entries.
        Optimized for constant-time performance relative to database size.
        
        Args:
            group_ids (Union[str, List[str]], optional): Single group ID or list of group IDs to delete
            data_sources (Union[str, List[str]], optional): Single data source or list of data sources to delete
                
        Returns:
            int: Number of rows deleted
                
        Raises:
            ValueError: If both group_ids and data_sources are None/empty
        """
        if not group_ids and not data_sources:
            raise ValueError("At least one of group_ids or data_sources must be provided")
        
        # Convert single values to lists for consistent handling
        if isinstance(group_ids, str):
            group_ids = [group_ids]
        if isinstance(data_sources, str):
            data_sources = [data_sources]
        
        with self.lock:
            try:
                self.conn.execute('BEGIN TRANSACTION;')
                
                # Cache index table names at initialization
                if not hasattr(self, '_index_tables'):
                    self.cursor.execute("""
                        SELECT name FROM sqlite_master 
                        WHERE type='table' 
                        AND (name LIKE 'index_forward_%' OR name LIKE 'index_backward_%')
                    """)
                    self._index_tables = [row[0] for row in self.cursor.fetchall()]
                
                # Build the WHERE clause dynamically based on provided parameters
                where_clauses = []
                params = []
                
                if group_ids:
                    placeholders = ','.join(['?' for _ in group_ids])
                    where_clauses.append(f'group_id IN ({placeholders})')
                    params.extend(group_ids)
                    
                if data_sources:
                    placeholders = ','.join(['?' for _ in data_sources])
                    where_clauses.append(f'data_source IN ({placeholders})')
                    params.extend(data_sources)
                    
                where_clause = ' OR '.join(where_clauses)
                
                # Process in batches to handle large deletions
                total_deleted = 0
                
                while True:
                    # Get next batch of UUIDs to delete using the indexed columns
                    self.cursor.execute(
                        f'SELECT uuid FROM main_table WHERE {where_clause} LIMIT {batch_size}',
                        params
                    )
                    batch_uuids = [row[0] for row in self.cursor.fetchall()]
                    
                    if not batch_uuids:
                        break
                    
                    # Delete from index tables using batched UUIDs
                    uuid_placeholders = ','.join(['?' for _ in batch_uuids])
                    for table in self._index_tables:
                        col_name = 'row_id'
                        self.cursor.execute(
                            f'DELETE FROM {table} WHERE {col_name} IN ({uuid_placeholders})',
                            batch_uuids
                        )
                    
                    # Delete from main table
                    self.cursor.execute(
                        f'DELETE FROM main_table WHERE uuid IN ({uuid_placeholders})',
                        batch_uuids
                    )
                    
                    total_deleted += len(batch_uuids)
                
                self.conn.commit()
                return total_deleted
                
            except Exception as e:
                self.conn.rollback()
                raise e

    def get_index_values_for_row(self, index_name, row_id):
        """
        Get all index values associated with a specific row.
        
        Args:
            index_name (str): Name of the index to search
            row_id (str): UUID of the row to look up
        
        Returns:
            list: List of index values associated with the row
        """
        self.cursor.execute(
            f'SELECT index_value FROM index_forward_{index_name} WHERE row_id = ?;',
            (row_id,)
        )
        values = self.cursor.fetchall()
        return [value[0] for value in values]
