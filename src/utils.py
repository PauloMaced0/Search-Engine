# Utility functions

import os
import msgpack

def print_short_index_entries(merged_index_file, min_length=200, max_items=15):
    """Print index entries with fewer than min_length items."""

    # Check if the file exists
    if not os.path.exists(merged_index_file):
        print(f"Error: File '{merged_index_file}' does not exist.")
        return

    # Open and read the merged index file using MessagePack
    with open(merged_index_file, 'rb') as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        merged_data = next(unpacker)

    short_entries = [
        (word, entries) for word, entries in merged_data['index'].items() 
        if len(entries) < min_length
    ]
    
    for word, entries in short_entries[:max_items]:
        print(f"{word}: {entries}")
