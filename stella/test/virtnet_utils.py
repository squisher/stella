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
from types import FunctionType
import time

class Settings(object):
    def setDefaults(self):
        self.settings = {
                'seed'      : [int(time.time()), int],
                'r'         : [0.1, float],
                'koff'      : [1.0, float],
                'radius'    : [10, int],
                'nlegs'     : [2, int],
                'gait'      : [2, int],
                'dim'       : [2, int],
                'nspiders'  : [1, int],     # not completely functional yet
                'elapsedTime':[self.elapsedTime, lambda x:x],
                }
    def elapsedTime(self):
        return time.time() - self.start_time

    def __init__(self, argv = []):
        self.start_time = time.time()

        self.setDefaults()

        if isinstance(argv, dict):
            for k, v in argv.items():
                self[k] = v
        else:
            # parse command line arguments to overwrite the defaults
            for key, _, val in [s.partition('=') for s in argv]:
                self[key] = val

    def __setitem__(self,k,v):
        if k in self.settings:
            self.settings[k][0] = self.settings[k][1](v)
        else:
            self.settings[k] = [v, type(v)]

    def __getitem__(self, k):
        return self.settings[k][0]

    def __str__(self):
        r = '{'
        for k,(v,type_) in self.settings.items():
            if isinstance(type_,  FunctionType):
                continue
            r += str(k) + ':' + str(v) + ', '
        return r[:-2] + '}'

