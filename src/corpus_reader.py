import ujson 

class CorpusReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_documents(self):
        """Reads the documents from the corpus and yields them one by one."""
        with open(self.file_path, 'r', buffering=1024 * 1024) as f:
            for line in f:
                yield ujson.loads(line)
