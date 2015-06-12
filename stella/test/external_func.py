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
from random import randint

import mtpy

from . import *  # noqa


def seed_const():
    mtpy.mt_seed32new(42)


def seed(s):
    mtpy.mt_seed32new(s)


def drand_const():
    mtpy.mt_seed32new(42)
    return mtpy.mt_drand()


def drand(s):
    mtpy.mt_seed32new(s)
    return mtpy.mt_drand() + mtpy.mt_drand()


@mark.parametrize('f', [seed_const, drand_const])
def test1(f):
    make_eq_test(f, ())


@mark.parametrize('arg', single_args([1, 2, 42, 1823828, randint(1, 10000000),
                                      randint(1, 10000000)]))
@mark.parametrize('f', [seed, drand])
def test2(f, arg):
    make_eq_test(f, arg)
