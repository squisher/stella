# Copyright 2013-2015 AP Ruymgaart, David Mohr
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
#######################################################################
# Heat equation finite difference
# visualize with Qt4
# AP Ruymgaart 4/14/2015
#######################################################################
gui = False

import time, sys, os, os.path
try:
    from . import unimplemented
except SystemError:
    def unimplemented(f):
        return f
try:
    from PySide.QtCore import *
    from PySide.QtGui import *
except ImportError:
    # fake stuff
    gui = False

    class QMainWindow(object):
        pass

import numpy as np


#######################################################################
"""
d/dx [ k(x) d/dx [u(x)] ] = d k(x)/dx * d u(x)/dx + k(x) Lapl u
d/dx [f g] = f'g + f g'

"""
#######################################################################

log = os.path.join(os.path.dirname(__file__), 'simulation.log')

def AppendLog(logfile, sz):
    f = open(logfile,"a+")
    f.write(sz)
    f.close()

#######################################################################
def WriteLog(sz):
    f = open(log,"a+")
    f.write(sz)
    f.close()

#######################################################################
def Diffusion2D(M, K, x,y, bPeriodic, gridsz):

    xSz = M.shape[1]
    ySz = M.shape[0]

    #-- boundary
    left = x-1
    right = x+1
    up = y+1
    down = y-1

    fl = 1.0
    fr = 1.0
    fu = 1.0
    fd = 1.0

    if (bPeriodic):
        #-- periodic boundary

        #-- x
        if (right >= xSz):
            right -= xSz
        if (left < 0):
            left += xSz

        #-- y
        if (up >= ySz):
            up -= ySz
        if (down < 0):
            down += ySz
    else:
        #-- Neumann (d2Tdxx )

        #-- x
        if (right >= xSz):
            right = left
            #fl = fr = 0.5
        if (left < 0):
            left = right
            #fl = fr = 0.5

        #-- y
        if (up >= ySz):
            up = down
            #fu = fd = 0.5
        if (down < 0):
            down = up
            #fu = fd = 0.5


    gridsz2 = gridsz * gridsz

    #-- diagonal of the Hessian:
    d2Tdxx = (fl*M[y,left] + fr*M[y,right] - 2.0*M[y,x]) / gridsz2
    d2Tdyy = (fu*M[up,x]   + fd*M[down,x]  - 2.0*M[y,x]) / gridsz2

    #-- Laplacian (trace of the Hessian)
    L = d2Tdxx + d2Tdyy

    #-- add the term coming from the variable k (gradient dot k)
    dot = 0.0

    dKdx = (K[y,left] - K[y,x])/gridsz
    dTdx = (M[y,left] - M[y,x])/gridsz

    dKdy = (K[up,x] - K[y,x])/gridsz
    dTdy = (M[up,x] - M[y,x])/gridsz

    dot = dKdx * dTdx + dKdy * dTdy

    du = K[y,x]*L + dot
    #if y == 10 and x == 9:
    #    print ("du={}".format(du))

    return du


########################################################################
class Frame(QMainWindow):
    def __init__(self, label, sim):

        self.mouseX = 0
        self.mouseY = 0

        super(Frame, self).__init__()
        label.setMouseTracking(True)
        QMainWindow.setCentralWidget(self, label)
        self.sim = sim

    def SetSizes(self, szx, szy):
        self.setGeometry(40, 40, self.sim.xSize*self.sim.scale + self.sim.border, self.sim.ySize*self.sim.scale + self.sim.border)

    #--------------------------
    def paintEvent(self, event):

        painter = QPainter(self)
        sz = "H=%10.1f time=%10.9f " % (self.sim.uTotal, self.sim.time)
        painter.drawText(15,15,sz)
        relX = self.mouseX - self.sim.border/2
        relY = self.mouseY - self.sim.border/2
        relX /= self.sim.scale
        relY /= self.sim.scale
        sz = "invalid point"
        if ((relX > 0) and (relX < self.sim.xSize)):
            if ((relY > 0) and (relY < self.sim.ySize)):
                sz = "x=%d y=%d   T=%9.8f     k=%9.8f    Src=%9.8f" % (relX, relY, self.sim.U[relY][relX], self.sim.K[relY][relX], self.sim.Source[relY][relX])
        painter.drawText(250,15,sz)

        for x in range(self.sim.xSize):
            for y in range(self.sim.ySize):

                col = self.HeatColor(self.sim.U[y,x])
                painter.fillRect(QRectF(x*self.sim.scale+self.sim.border/2, y*self.sim.scale+self.sim.border/2, self.sim.scale, self.sim.scale), col)

    #--------------------------
    # NOTE: NEED + and - limits (max,min) to be symmetric
    #-- zero = black
    #-- below zero, blue (vary brightness)
    #-- from zero up, transition from black to green to red
    #--
    def HeatColor(self, u):

        #intvl = [0, self.uMax]
        #intvl8bit = [0,255]

        R = 0
        G = 0
        B = 0

        t = abs(u)
        v8bit = t * 10
        if (v8bit > 255):
            v8bit = 255

        G = 255 - abs(v8bit)

        if (u > 0):
            R = v8bit
            B = G
        else:
            B = v8bit
            R = G


        return QColor(R,G,B)

    #--------------------------
    def run(self):
        """
        HACK: instead of passing in a function to sim.run(), this is an adoptation of sim.run() for
        the gui.
        """

        for n in range(self.sim.nsteps):

            self.sim.GetHeat(n * self.sim.paintsteps)
            self.update()
            QApplication.processEvents()
            time.sleep(0.01)

        print ("Done.")

    def run_no_sleep(self):
        """
        HACK: instead of passing in a function to sim.run(), this is an adoptation of sim.run() for
        the gui.
        """

        for n in range(self.sim.nsteps):

            self.sim.GetHeat(n * self.sim.paintsteps)
            self.update()
            QApplication.processEvents()

        print ("Done.")

    #--------------------------
    def mouseMoveEvent(self, event):

        self.mouseX = event.x()
        self.mouseY = event.y()


class Sim(object):
    def __init__(self):
        self.xSize = 100
        self.ySize = 100

        self.xm = 1.95
        self.ym = 0.3
        self.gridSize = self.xm/self.xSize
        self.gridArea = self.gridSize*self.gridSize
        self.gridVolume = self.gridSize*self.gridSize*self.gridSize
        self.sourceVolume = 0.0

        self.U = np.zeros((1, 1))
        self.Source = np.zeros((1, 1))
        self.Sink = np.zeros((1, 1))
        self.K = np.zeros((1, 1))

        self.src = 1.0
        self.sinktemp = -5.0

        self.scale = 4

        self.dt = 0.04
        self.set_nsteps(2000)
        self.paintsteps = 10
        self.time = 0.0

        self.border = 100

        self.uMax = 100.0
        self.uMin = -100.0
        self.uRange = self.uMax #- self.uMin
        self.uTotal = 0.0

        self.bPeriodic = True


    def set_nsteps(self, nsteps):
        self.nsteps = nsteps
        self.observations = np.zeros((nsteps, 5))

    #--------------------------
    def SetSizes(self, szx, szy, p=False):

        self.xSize = szx
        self.ySize = szy

        self.U = np.zeros((self.ySize, self.xSize))
        self.Source = np.zeros((self.ySize, self.xSize))
        self.Sink = np.zeros((self.ySize, self.xSize))
        self.K = np.zeros((self.ySize, self.xSize))

        self.xm = float(self.xSize) * 0.003
        self.ym = float(self.ySize) * 0.003

        self.gridSize = self.xm/self.xSize
        self.gridArea = self.gridSize*self.gridSize
        self.gridVolume = self.gridSize*self.gridSize*self.gridSize


        if abs(self.gridSize - self.ym/self.ySize) > 0.00001:
            print ("SIZE ERROR, grid not square", self.xm, self.ym, self.ym/self.ySize, self.gridSize)
            exit()

        sz =    """
################################# SIZES ############################
x=%4.1f, y=%4.1f (m)
x=%5d, y=%5d (tiles)
grid cell size=%f (m)
grid cell area=%f (m^2) grid cell volume=%10.9f (m^3)
total volume=%f (m^3)
""" % (self.xm, self.ym, self.xSize, self.ySize, self.gridSize, self.gridArea, self.gridVolume,self.gridVolume*self.xSize*self.ySize)
        if p:
            print (sz)
        WriteLog(sz)


        sz =    """
########################### INTEGRATION ############################
number steps=%d
timestep=%f (s)
""" % (self.nsteps, self.dt)
        if p:
            print (sz)
        WriteLog(sz)


        sz =    """
############################## START ###############################
"""
        WriteLog(sz)


    #--------------------------
    def GetHeat(self, n):

        #--
        self.uTotal = 0.0
        self.uMax = -10000.0
        self.uMin = -1.0 * self.uMax

        for x in range(self.xSize):
            for y in range(self.ySize):

                du = Diffusion2D(self.U, self.K, x,y, self.bPeriodic, self.gridSize)
                self.uTotal += self.U[y,x]

                #-- timestep
                self.U[y,x] +=  du * self.dt

                #-- Sources and Sinks
                self.U[y,x] += self.Source[y,x] * self.dt
                self.U[y,x] -= self.Sink[y,x] * self.dt


                if (self.U[y,x] > self.uMax):
                    self.uMax = self.U[y,x]
                if (self.U[y,x] < self.uMin):
                    self.uMin = self.U[y,x]

        self.time = self.dt * float(n)
        #sz = "%08d %12.11f %12.11f %4.1f %4.1f\n" % (n, self.time, self.uTotal, self.uMax, self.uMin)
        #print (sz)
        #AppendLog(sz)


    #--------------------------
    def run(self):

        for n in range(self.nsteps):

            self.GetHeat(n * self.paintsteps)
            self.observations[n, 0] = n
            self.observations[n, 1] = self.time
            self.observations[n, 2] = self.uTotal
            self.observations[n, 3] = self.uMax
            self.observations[n, 4] = self.uMin


def process_config(sim, settings="heat_settings.txt", p=False):
    #---- VERY SIMPLE KEYWORD PARSER ----
    inFile = open(os.path.join(os.path.dirname(__file__), settings),"r")
    data = inFile.readlines()
    inFile.close()
    for line in data:
        elms = line.split()
        if (len(elms)):
            if (elms[0][0] != '#'):
                key = elms[0]
                if key == 'dt':
                    sim.dt = float(elms[2])
                    if p:
                        print ("SETTING dt", sim.dt)

                if key == 'steps':
                    sim.set_nsteps(int(elms[2]))
                    if p:
                        print ("SETTING steps", sim.nsteps)

                if key == 'paintsteps':
                    sim.paintsteps = int(elms[2])
                    if p:
                        print ("SETTING paintsteps", sim.paintsteps)

                if key == 'src':
                    sim.src = float(elms[2])
                    if p:
                        print ("SETTING src (K/s)", sim.src)

                if key == 'sinktemp':
                    sim.sinktemp = float(elms[2])
                    if p:
                        print ("SETTING sinktemp ", sim.sinktemp)


                if key == 'scale':
                    sim.scale = int(elms[2])
                    if p:
                        print ("SETTING scale", sim.scale)


                if key == 'grid':
                    sim.xSize = int(elms[2])
                    sim.ySize = int(elms[3])
                    sim.SetSizes(sim.xSize, sim.ySize, p)
                    for y in range(sim.ySize):
                        for x in range(sim.xSize):
                            sim.K[y, x] = 1.0
                    if p:
                        print ("SETTING grid", sim.xSize, sim.ySize)


                if key == 'source':
                    sim.Source[int(elms[2])][int(elms[1])] = float(elms[3])

                if key == 'temp':
                    sim.U[int(elms[2])][int(elms[1])] = float(elms[3])

                if key == 'alpha':
                    sim.K[int(elms[2])][int(elms[1])] = float(elms[3])


def test1():
    import stella

    sim1 = Sim()
    sim2 = Sim()

    settings = 'test1_settings.txt'
    process_config(sim1, settings)
    process_config(sim2, settings)

    sim1.run()
    stella.wrap(sim2.run)()

    print ("Python:")
    format_result(sim1)
    print ("Stella:")
    format_result(sim2)

    assert((sim1.observations == sim2.observations).all())


def format_result(sim):
    for n, time_, uTotal, uMax, uMin in sim.observations:
        #sz = "%08d %12.11f %12.11f %4.1f %4.1f\n" % (n, self.time, self.uTotal, self.uMax, self.uMin)
        #print ("%08d %12.11f %12.11f %4.1f %4.1f" % (n, time_, uTotal, uMax, uMin))
        print ("%016.8f %12.11f %12.11f %12.8f %12.8f" % (n, time_, uTotal, uMax, uMin))


def prepare(args):
    sim = Sim()
    process_config(sim)
    sim.set_nsteps(args['nsteps'])

    def get_result(r):
        format_result(sim)
        return sim.observations

    return (sim.run, (), get_result)


########################################################################
if __name__ == '__main__':
    sim = Sim()
    if gui:
        example = QApplication(sys.argv)
        label = QLabel()
        frm = Frame(label, sim)
        frm.setMouseTracking(True)

        frm.show()
        frm.raise_()

    if len(sys.argv) > 1:
        settings = sys.argv[1]
    else:
        settings = 'test1_settings.txt'
    process_config(sim, settings, p=False)

    if gui:
        frm.SetSizes(sim.xSize, sim.ySize)
        frm.run_no_sleep()
        sys.exit(example.exec_())
    else:
        sim.run()
        format_result(sim)

# pylama:ignore=E401,W0401,E302,E231,E265,E303,E221,E261,E262,E501,E211,E222
