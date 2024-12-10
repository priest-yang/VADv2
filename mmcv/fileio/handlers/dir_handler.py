import os
import sqlite3
import pickle
import numpy as np
from .base import BaseFileHandler
from .db_handler import SqliteDictWrapper
from .pickle_handler import PickleHandler

class DirWrapper:
    def __init__(self, dir_path):
        """
        Initializes the wrapper by scanning the directory for SQLite databases.

        Args:
            dir_path (str): Path to the directory containing SQLite database files.
        """
        self.databases = []
        self.lengths = []
        self.lookup_table = {}
        self.total_length = 0

        for file in sorted(os.listdir(dir_path)):
            if file.endswith('.sqlite') or file.endswith('.db'):
                db_path = os.path.join(dir_path, file)
                db_wrapper = SqliteDictWrapper(db_path)
                self.databases.append(db_wrapper)
                self.lengths.append(len(db_wrapper))
                self.total_length += len(db_wrapper)
            elif file.endswith('.pkl'):
                db_path = os.path.join(dir_path, file)
                pkl_wrapper = PickleHandler().load_from_path(db_path)
                if isinstance(pkl_wrapper, dict) and "infos" in pkl_wrapper:
                    pkl_wrapper = pkl_wrapper["infos"]
                self.databases.append(pkl_wrapper)
                self.lengths.append(len(pkl_wrapper))
                self.total_length += len(pkl_wrapper)

        # pesudo metadata
        self.metadata = {'version': '1.0-trainval'}
    
    def _get_metadata(self):
        return self.metadata

    def __getitem__(self, idx):
        """
        Retrieves the item(s) at the specified index or indices from the combined databases.

        Args:
            idx (int, list, or tuple): The index or indices of the item(s) to retrieve.

        Returns:
            dict or list of dict: The deserialized dictionary or list of dictionaries from the appropriate database.
        
        Raises:
            IndexError: If any index is out of range.
            TypeError: If the index type is unsupported.
        """
        if idx == 'metadata':
            return self._get_metadata()
        
        if isinstance(idx, (list, tuple)):
            return [self._get_item(i) for i in idx]
        elif isinstance(idx, (int, np.integer)):
            return self._get_item(idx)
        else:
            raise TypeError('Indices must be integers or lists/tuples of integers')

    def _get_item(self, idx):
        """Helper method to retrieve a single item by index."""
        if idx < 0 or idx >= self.total_length:
            raise IndexError('Index out of range')
        
        # Determine the appropriate database and adjusted index
        db_index = 0
        for length in self.lengths:
            if idx < length:
                return self.databases[db_index][idx]
            idx -= length
            db_index += 1

    def __len__(self):
        """
        Returns the total number of items across all databases.

        Returns:
            int: Total number of items.
        """
        return self.total_length

    def close(self):
        """
        Closes all database connections.
        """
        for obj in self.databases:
            if isinstance(obj, SqliteDictWrapper):
                obj.close()
            elif isinstance(obj, (list, dict, tuple)):
                del obj

    def __del__(self):
        """
        Ensures all database connections are closed when the object is deleted.
        """
        self.close()


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

class DirHandler(BaseFileHandler):
    """
    Directory handler for reading and writing data to multiple SQLite databases.
    """
    str_like = False

    def load_from_fileobj(self, file):
        """
        Loads data from a file-like object.

        Args:
            file (file-like object): The file-like object to read from.

        Returns:
            DirWrapper: A wrapper around the directory containing SQLite databases.
        """
        return DirWrapper(file.name)

    def load_from_path(self, filepath):
        """
        Loads data from a file path.

        Args:
            filepath (str): The path to the directory containing SQLite databases.

        Returns:
            DirWrapper: A wrapper around the directory containing SQLite databases.
        """
        return DirWrapper(filepath)

    def dump_to_fileobj(self, obj, file):
        """
        Dumps data to a file-like object.

        Args:
            obj (DirWrapper): The DirWrapper object to dump.
            file (file-like object): The file-like object to write to.
        """
        raise NotImplementedError('Dumping to a file-like object is not supported')

    def dump_to_str(self, obj):
        """
        Not implemented because dumping to a string is not applicable for directories.
        """
        raise NotImplementedError('Dumping to a string is not supported')

    def dump_to_path(self, obj, filepath):
        """
        Dumps data to a file path.

        Args:
            obj (DirWrapper): The DirWrapper object to dump.
            filepath (str): The path to the directory to write to.
        """
        raise NotImplementedError('Dumping to a file path is not supported')