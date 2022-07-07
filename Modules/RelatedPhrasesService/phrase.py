class Phrase:
    def __init__(self, text: str, document: int, position: int, is_interesting: bool = False):
        self.text: str = text
        self.status: str = "possible_phrase"  # Stage
        self.bad_status: str = ""
        self.p: int = 1  # Count of documents has included
        self.s: int = 1  # Count of all including phrase
        self.m: int = 0  # Count of interesting including
        self.e: float = None
        self.predicts: bool = False
        self.can_predict: set = set()

        self.position_in_documents: dict = dict()   # Documents including with positions
        self.documents_index: dict = None

        self.position_in_documents[document] = [position]

        if is_interesting:
            self.m += 1

    # Add phrase from document
    def add(self, document: int, position: int, is_interesting: bool = False):
        if document not in self.position_in_documents.keys():
            self.position_in_documents[document] = [position]
        else:
            self.position_in_documents[document].append(position)

        self.p = len(self.position_in_documents)
        self.s += 1
        if is_interesting:
            self.m += 1

    def __iadd__(self, other):
        self.s += other.s

        # Duplicates documents add
        duplicate_keys_documents = (
            set(
                other.position_in_documents.keys()
            ).intersection(
                self.position_in_documents.keys()
            )
        )

        for document_key in duplicate_keys_documents:
            self.position_in_documents[document_key] += \
                other.position_in_documents[document_key]
            del other.position_in_documents[document_key]

        # Other documents add
        for document_key in other.position_in_documents.keys():
            self.position_in_documents[document_key] = \
                other.position_in_documents[document_key]

        self.p = len(self.position_in_documents)
        return self

    def __str__(self):
        e = '{0:7.3f}'.format(self.e) if self.e is not None else '   None'
        return "{0:70}\t {1:5} \t {2:5} \t {3} \t {4:14} \t {5} \t {6}".format(self.text, self.p, self.s, e,
                                                                               self.status, self.position_in_documents,
                                                                               self.bad_status)
