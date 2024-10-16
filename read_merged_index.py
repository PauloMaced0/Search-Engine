import msgpack
import os

def read_merged_index(merged_index_file):
    # Check if the file exists
    if not os.path.exists(merged_index_file):
        print(f"Error: File '{merged_index_file}' does not exist.")
        return

    # Open and read the merged index file using MessagePack
    with open(merged_index_file, 'rb') as f:
        merged_data = msgpack.unpack(f)

    # Display the inverted index
    # for word in merged_data['index'].items():
    #     print(word)
    #     return

    for word in merged_data['doc_lengths'].items():
        print(word)
        return

if __name__ == "__main__":
    # Specify the path to the merged index file
    merged_index_file = 'output/merged_index.msgpack'

    # Read and display the merged index
    read_merged_index(merged_index_file)
