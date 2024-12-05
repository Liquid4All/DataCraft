import sqlite3
import os
import uuid
import asyncio
import threading
import inspect
from typing import List, Dict, Any, Optional, Union, Callable

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

class DataManager(SQLBaseManager):
    """
    A class for managing data storage and retrieval using SQLite with support for
    batch operations, custom indexing, and async processing.
    
    Attributes:
        db_path (str): Path to the SQLite database file
        conn (sqlite3.Connection): Database connection object
        cursor (sqlite3.Cursor): Database cursor object
    """

    def __init__(self, db_path: str):
        """
        Initialize the DataManager with a database path.

        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self.lock = threading.Lock()
        self._connect_db()
        self._create_main_table()

    def insert_data(self, entries: Union[Dict[str, Any], List[Dict[str, Any]]]):
        """
        Insert one or more data entries into the database.

        Args:
            entries: Single dictionary or list of dictionaries containing:
                - data_source (str): Source of the data
                - group_id (str): Group identifier
                - data_type (str): Type of data
                - data (str): Actual data content
                - files (list): List of associated file paths
                - uuid_str (str, optional): Custom UUID

        Returns:
            list: List of UUIDs for inserted entries
        """
        with self.lock:
            if not isinstance(entries, list):
                entries = [entries]

            insert_sql = '''
            INSERT INTO main_table (uuid, data_source, group_id, data_type, data, files)
            VALUES (?, ?, ?, ?, ?, ?);
            '''
            
            data_list = []
            uuids = []
            
            for entry in entries:
                uuid_str = entry.get('uuid_str', str(uuid.uuid4()))
                uuids.append(uuid_str)
                files_str = ','.join(entry['files'])
                
                data_list.append((
                    uuid_str,
                    entry['data_source'],
                    entry['group_id'],
                    entry['data_type'],
                    entry['data'],
                    files_str
                ))

            self.conn.execute('BEGIN TRANSACTION;')
            try:
                self.cursor.executemany(insert_sql, data_list)
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                raise e

            return uuids

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

    def delete_group(self, group_id):
        """
        Delete all data associated with a specific group ID.
        
        Args:
            group_id (str): Group ID to delete
            
        Returns:
            int: Number of records deleted
        """
        with self.lock:
            self.cursor.execute("DELETE FROM main_table WHERE group_id = ?", (group_id,))
            deleted_count = self.cursor.rowcount
            self.conn.commit()
            return deleted_count

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
