class Phrase:
    def __init__(self, text: str, document: int, is_interesting: bool = False):
        self.text: str = ""
        self.status: str = "possible_phrase"  # Stage
        self.bad_status: str = ""
        self.p: int = 1  # Count of documents has included
        self.s: int = 1  # Count of all including phrase
        self.m: int = 0  # Count of interesting including
        self.e: float = None
        self.predicts: bool = False
        self.can_predict: set = set()

        self.documents_s: dict = dict()      # Documents including
        self.documents_index: dict = None

        self.text = text
        self.documents_s[document] = 1
        if is_interesting:
            self.m += 1

    # Add phrase from document
    def add(self, document: int, is_interesting: bool = False):
        if document not in self.documents_s.keys():
            self.documents_s[document] = 1
        else:
            self.documents_s[document] += 1

        self.p = len(self.documents_s)
        self.s += 1
        if is_interesting:
            self.m += 1

    def __str__(self):
        e = '{0:7.3f}'.format(self.e) if self.e is not None else '   None'
        return "{0:70}\t {1:5} \t {2:5} \t {3} \t {4:14} \t {5} \t {6}".format(self.text, self.p, self.s, e,
                                                                               self.status, self.documents_s,
                                                                               self.bad_status)
