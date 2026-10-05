class Candidate:

    def __init__(self, operations=None):
        self.operations = operations or []


    def __repr__(self):
        return f"Candidate(operations={self.operations})"