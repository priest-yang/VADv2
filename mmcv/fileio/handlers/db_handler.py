import sqlite3
import pickle
from .base import BaseFileHandler
import os
# list like wrapper for sqlite dict
class SqliteDictWrapper:
    def __init__(self, sqlite_db_path):
        """
        Initializes the wrapper by connecting to the SQLite database.

        Args:
            sqlite_db_path (str): Path to the SQLite database file.
        """
        if not os.path.exists(sqlite_db_path):
            raise FileNotFoundError(f"{sqlite_db_path} Not Found")

        self.conn = sqlite3.connect(sqlite_db_path)
        self.cursor = self.conn.cursor()
        # Get the total number of rows for __len__
        self.cursor.execute("SELECT COUNT(1) FROM infos")  # use autoincrement seq to get length, could be faster, but not guaranteed if rows are deleted
        self._length = self.cursor.fetchone()[0]

        # pesudo metadata
        self.metadata = {'version': '1.0-trainval'}

    
    def _get_metadata(self):
        return self.metadata


    def __getitem__(self, idx):
        """
        Retrieves the item(s) at the specified index or indices from the database.

        Args:
            idx (int or list/tuple of int): Index or indices of the item(s) to retrieve.

        Returns:
            dict or list of dict: The deserialized dictionary or list of dictionaries from the database.

        Raises:
            IndexError: If any index is out of range.
            TypeError: If the index type is unsupported.
        """
        if idx == 'metadata':
            return self._get_metadata()
            
        if isinstance(idx, (int, np.integer)):
            # Handle single integer index
            return self._get_item_by_index(idx)
        elif isinstance(idx, (list, tuple)):
            # Handle list or tuple of indices
            return self._get_items_by_indices(idx)
        else:
            raise TypeError('Indices must be integers or lists/tuples of integers')

    def _get_item_by_index(self, idx):
        """Helper method to retrieve a single item by index."""
        idx = int(idx)
        if idx < 0:
            idx = self._length + idx  # Handle negative indices
        if idx < 0 or idx >= self._length:
            raise IndexError('Index out of range')
        # Fetch the data at the given index
        self.cursor.execute('SELECT data FROM infos WHERE id = ?', (idx + 1,))  # IDs start from 1
        row = self.cursor.fetchone()
        if row:
            serialized_data = row[0]
            data = pickle.loads(serialized_data)
            return data
        else:
            raise IndexError('Index out of range')

    def _get_items_by_indices(self, indices):
        """Helper method to retrieve multiple items by indices."""
        # Handle negative indices and validate
        valid_indices = []
        for idx in indices:
            if not isinstance(idx, int):
                raise TypeError('All indices must be integers')
            if idx < 0:
                idx = self._length + idx  # Handle negative indices
            if idx < 0 or idx >= self._length:
                raise IndexError(f'Index out of range: {idx}')
            valid_indices.append(idx + 1)  # Adjust for 1-based IDs

        # Build the query with placeholders
        placeholders = ','.join('?' * len(valid_indices))
        query = f'SELECT id, data FROM infos WHERE id IN ({placeholders})'
        self.cursor.execute(query, valid_indices)
        rows = self.cursor.fetchall()

        # Map IDs to data for ordering
        id_to_data = {row[0]: pickle.loads(row[1]) for row in rows}

        # Return data in the order of requested indices
        result = [id_to_data[idx + 1] for idx in indices]
        return result

    def __len__(self):
        """
        Returns the total number of items in the database.

        Returns:
            int: Total number of items.
        """
        return self._length

    def close(self):
        """
        Closes the database connection.
        """
        self.conn.close()

    def __del__(self):
        """
        Ensures the database connection is closed when the object is deleted.
        """
        self.close()



import sqlite3
import pickle
from .base import BaseFileHandler
import numpy as np

def set_default(obj):
    """Set default json values for non-serializable values.

    It helps convert ``set``, ``range`` and ``np.ndarray`` data types to list.
    It also converts ``np.generic`` (including ``np.int32``, ``np.float32``,
    etc.) into plain numbers of plain python built-in types.
    """
    if isinstance(obj, (set, range)):
        return list(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f'{type(obj)} is unsupported for json dump')




class DbHandler(BaseFileHandler):

    str_like = False

    def load_from_fileobj(self, file):
        """
        Not implemented because SQLite databases require file paths,
        and handling file-like objects is not suitable for this handler.
        """
        raise NotImplementedError("load_from_fileobj is not implemented for DbHandler.")

    def dump_to_fileobj(self, obj, file):
        """
        Not implemented because SQLite databases require file paths,
        and handling file-like objects is not suitable for this handler.
        """
        raise NotImplementedError("dump_to_fileobj is not implemented for DbHandler.")

    def dump_to_str(self, obj):
        """
        Not implemented because dumping to a string is not applicable for SQLite databases.
        """
        raise NotImplementedError("dump_to_str is not implemented for DbHandler.")

    def load_from_path(self, filepath):
        """
        Load data from a SQLite database at the given filepath.
        Return a list containing all the data.
        """
        return SqliteDictWrapper(filepath)

    def dump_to_path(self, obj, filepath):
        """
        Dump data to a SQLite database at the given filepath.
        obj should be an iterable of dictionaries or serializable objects.
        """
        # Create a connection to the SQLite database
        conn = sqlite3.connect(filepath)
        cursor = conn.cursor()

        # Create the table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS infos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data BLOB
            )
        ''')

        # Clear existing data
        cursor.execute('DELETE FROM infos')

        obj = set_default(obj)

        # Insert data into the database
        for item in obj:
            # Serialize the item using pickle
            serialized_data = pickle.dumps(item)
            cursor.execute('INSERT INTO infos (data) VALUES (?)', (serialized_data,))

        # Commit changes and close the connection
        conn.commit()
        conn.close()

