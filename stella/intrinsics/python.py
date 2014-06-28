def zeros(shape=1, dtype=None):
    """
    Emulate certain features of `numpy.zeros`

    Note:
    * `dtype` is ignored in Python, but will be interpreted in Stella.
    * This is for testing only! Memory allocation (and deallocation) is not
      a feature of Stella at this point in time.
    """
    try:
        dim = len(shape)
        if dim == 1:
            shape = shape[0]
            raise TypeError()
    except TypeError:
        return [0 for i in range(shape)]

    # here dim > 1, build up the inner most dimension
    inner = [0 for i in range(shape[dim-1])]
    for i in range(dim-2, -1, -1):
        new_inner = [list(inner) for j in range(shape[i])]
        inner = new_inner
    return inner
