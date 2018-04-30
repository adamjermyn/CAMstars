class CaseInsensitiveKey(object):
    def __init__(self, key):
        self.key = key
    def __hash__(self):
        return hash(self.key.lower())
    def __eq__(self, other):
        return self.key.lower() == other.key.lower()
    def __str__(self):
        return self.key
class CaseInsensitiveDict(dict):
    def __setitem__(self, key, value):
        key = CaseInsensitiveKey(key)
        super(CaseInsensitiveDict, self).__setitem__(key, value)
    def __getitem__(self, key):
        key = CaseInsensitiveKey(key)
        return super(CaseInsensitiveDict, self).__getitem__(key)