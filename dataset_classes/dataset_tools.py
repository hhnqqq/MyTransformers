import subprocess
from mmap import mmap

def get_line_count(file_path):
    result = subprocess.run(['wc', '-l', file_path], capture_output=True, text=True)
    return int(result.stdout.split()[0])

class MmapDataset:
    """A dataset class that uses memory-mapped file for efficient line access.


    Attributes:
        filename (str): The path to the file to be memory-mapped.
        file (file object): The file object opened in read-binary mode.
        mm (mmap.mmap): The memory-mapped file object.
        line_offsets (list of int): The list of offsets for the start of each line in the file.
    """
    def __init__(self, filename):
        """Initializes MmapDataset with the given filename.
        
        Args:
            filename (str): The path to the file to be memory-mapped.
        """
        self.filename = filename
        self.file = open(filename, 'r+b')
        self.mm = mmap(self.file.fileno(), 0)
        self.line_offsets = self._index_lines()

    def _index_lines(self):
        """Indexes the starting offset of each line in the memory-mapped file.
    
        Returns:
            list of int: The list of offsets for the start of each line in the file.
        """
        offsets = [0]
        self.mm.seek(0)
        for line in iter(self.mm.readline, b''):
            offsets.append(self.mm.tell())
        return offsets[:-1]

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.line_offsets):
            raise IndexError("Index out of range")
        self.mm.seek(self.line_offsets[idx])
        return self.mm.readline().decode().strip()

    def __len__(self):
        return len(self.line_offsets)

    def __del__(self):
        self.mm.close()
        self.file.close()