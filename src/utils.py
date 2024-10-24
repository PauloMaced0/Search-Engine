# Utility functions
import os
import msgpack

def read_merged_index(merged_index_file):
    # Check if the file exists
    if not os.path.exists(merged_index_file):
        print(f"Error: File '{merged_index_file}' does not exist.")
        return

    # Open and read the merged index file using MessagePack
    with open(merged_index_file, 'rb') as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        merged_data = next(unpacker)

    # Display the inverted index

    # for word in merged_data['index'].items():
    #     print(word)
    #     return

    # Display the document lengths 

    for word in merged_data['doc_lengths'].items():
        print(word)
        return

