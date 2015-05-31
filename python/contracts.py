def check(condition, error_message=''):
    if not condition:
        raise AssertionError(error_message)