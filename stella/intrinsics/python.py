# Copyright 2013-2015 David Mohr
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
