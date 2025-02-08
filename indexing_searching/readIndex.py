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

    i = 0
    for word in merged_data['index'].items():
        if len(word[1]) < 200:
            print(word)
            i = i + 1
            if i > 15:
                return 

    # Display the document lengths 

    # i = 0
    # for word in merged_data['doc_lengths'].items():
    #     print(word)
    #     i = i + 1
    #     if i > 10:
    #         return 

def main():
    read_merged_index("output/merged_index.msgpack")

if __name__ == '__main__':
    main()
