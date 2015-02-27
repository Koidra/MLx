def check(condition, error_message=''):
    if not condition:
        raise AssertionError(error_message)

# Extension of dict
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
