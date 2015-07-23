#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#define True 1
#define False 0
#define bool uint8_t

#define sfs_get(s,x,y) \
    ((s)[(x)*dim_x + (y)])

double* idx(double *a, int x, int y, int sx, int sy) {
    //return a+(y*sy+x);
    return &a[x*sy+y];
}
double get(double *a, int x, int y, int sx, int sy) {
    return *idx(a, x, y, sx, sy);
}
double set(double *a, int x, int y, int sx, int sy, double v) {
    return (*idx(a, x, y, sx, sy) = v);
}

typedef struct {
    int xSize;
    int ySize;

    double xm;
    double ym;
    double gridSize;
    double gridArea;
    double gridVolume;
    //int sourceVolume;

    double *U;
    double *Source;
    double *Sink;
    double *K;

    double src;
    double sinktemp;

    int scale;

    double dt;
    int paintsteps;
    double time;

    int nsteps;
    double *observations;

    int border;

    double uMax;
    double uMin;
    double uRange;
    double uTotal;

    bool bPeriodic;
} tSim;

double * np_array(int szx, int szy, double init)
{
    double *r;
    int sz;

    sz = szx * szy * sizeof(double);
    r = (double *) malloc(sz);
    if (init == 0)
        memset(r, '\0', sz);
    else for (int i=0; i<szx; i++)
            for (int j=0; j<szy; j++)
                set(r, i, j, szx, szy, init);
    return r;
}
#define np_zeros(szx, szy) np_array(szx, szy, 0)
#define np_ones(szx, szy) np_array(szx, szy, 1)

/* *-*-* */

double Diffusion2D(double *M, double *K, int x, int y, int xSz, int ySz, bool bPeriodic, double gridsz)
{
    //-- boundary
    int left = x-1;
    int right = x+1;
    int up = y+1;
    int down = y-1;

    double fl = 1.0;
    double fr = 1.0;
    double fu = 1.0;
    double fd = 1.0;

    if (bPeriodic) {
        //-- periodic boundary

        //-- x
        if (right >= xSz)
            right -= xSz;
        if (left < 0)
            left += xSz;

        //-- y
        if (up >= ySz)
            up -= ySz;
        if (down < 0)
            down += ySz;
    } else {
        //-- Neumann (d2Tdxx )

        //-- x
        if (right >= xSz)
            right = left;
            //fl = fr = 0.5
        if (left < 0)
            left = right;
            //fl = fr = 0.5

        //-- y
        if (up >= ySz)
            up = down;
            //fu = fd = 0.5
        if (down < 0)
            down = up;
            //fu = fd = 0.5
    }

    double gridsz2 = gridsz * gridsz;

    //-- diagonal of the Hessian:
    double d2Tdxx = (fl*get(M, y,left, xSz, ySz) + fr*get(M, y,right, xSz, ySz) - 2.0*get(M, y,x, xSz, ySz)) / gridsz2;
    double d2Tdyy = (fu*get(M, up,x, xSz, ySz)   + fd*get(M, down,x, xSz, ySz)  - 2.0*get(M, y,x, xSz, ySz)) / gridsz2;

    //-- Laplacian (trace of the Hessian)
    double L = d2Tdxx + d2Tdyy;

    //-- add the term coming from the variable k (gradient dot k)
    double dot = 0.0;

    double dKdx = (get(K, y,left, xSz, ySz) - get(K, y,x, xSz, ySz))/gridsz;
    double dTdx = (get(M, y,left, xSz, ySz) - get(M, y,x, xSz, ySz))/gridsz;

    double dKdy = (get(K, up,x, xSz, ySz) - get(K, y,x, xSz, ySz))/gridsz;
    double dTdy = (get(M, up,x, xSz, ySz) - get(M, y,x, xSz, ySz))/gridsz;

    dot = dKdx * dTdx + dKdy * dTdy;

    double du = get(K, y,x, xSz, ySz)*L + dot;

    /*
    if (y == 10 && x == 9) {
        printf ("du=%f\n", du);
    }
    */

    return du;
}

void set_nsteps(tSim *self, int nsteps)
{
    self->nsteps = nsteps;
    self->observations = np_zeros(nsteps, 5);
}

void SetSizes(tSim *self, int szx, int szy)
{
   self->xSize = szx;
   self->ySize = szy;

   self->U = np_zeros(self->ySize, self->xSize);
   self->Source = np_zeros(self->ySize, self->xSize);
   self->Sink = np_zeros(self->ySize, self->xSize);
   self->K = np_ones(self->ySize, self->xSize);

   self->xm = ((float)self->xSize) * 0.003;
   self->ym = ((float)self->ySize) * 0.003;

   self->gridSize = self->xm/self->xSize;
   self->gridArea = self->gridSize*self->gridSize;
   self->gridVolume = self->gridSize*self->gridSize*self->gridSize;

   if (fabs(self->gridSize - self->ym/self->ySize) > 0.00001) {
       printf ("SIZE ERROR, grid not square %f %f %f %f\n", self->xm, self->ym, self->ym/self->ySize, self->gridSize);
       exit(1);
   }
}

void __init__(tSim *self)
{
    SetSizes(self, 100, 100);
 
    /*
    self->sourceVolume = 0.0;
 
    self->U = np.zeros((1, 1));
    self->Source = np.zeros((1, 1));
    self->Sink = np.zeros((1, 1));
    self->K = np.zeros((1, 1));
    */
    int {{nsteps_init}};
 
    self->dt = 0.000000002; // 0.04;
    set_nsteps(self, nsteps);
    self->paintsteps = 1000; // 10;
    self->src = 0.4255; // 1.0;
    self->sinktemp = -15; // -5.0;
    self->scale = 4;
 
    /* note that the original source uses always the coordinates 'y, x' */
    set(self->Source, 10, 10, self->xSize, self->ySize, 10000000.0);
    set(self->Source, 40, 10, self->xSize, self->ySize, -10000000.0);
    set(self->Source, 26, 65, self->xSize, self->ySize, 10000000.0);
    set(self->Source, 36, 80, self->xSize, self->ySize, -10000000.0);
    // temp
    set(self->U, 10, 10, self->xSize, self->ySize, 50);
    set(self->U, 40, 40, self->xSize, self->ySize, -50);
    // alpha         
    set(self->K, 10, 10, self->xSize, self->ySize, 30);
    set(self->K, 10, 11, self->xSize, self->ySize, 30);
    set(self->K, 10, 12, self->xSize, self->ySize, 30);
    set(self->K, 11, 10, self->xSize, self->ySize, 30);
    set(self->K, 11, 11, self->xSize, self->ySize, 30);
    set(self->K, 11, 12, self->xSize, self->ySize, 30);
    set(self->K, 12, 10, self->xSize, self->ySize, 30);
    set(self->K, 12, 11, self->xSize, self->ySize, 30);
    set(self->K, 12, 12, self->xSize, self->ySize, 30);
 
    self->time = 0.0;
    self->border = 100;
    self->uMax = 100.0;
    self->uMin = -100.0;
    self->uRange = self->uMax; //- self->uMin;
    self->uTotal = 0.0;
    self->bPeriodic = True;
}

void GetHeat(tSim *self, int n)
{
    //--
    self->uTotal = 0.0;
    self->uMax = -10000.0;
    self->uMin = -1.0 * self->uMax;

    for (int x=0; x<self->xSize; x++) {
        for (int y=0; y<self->ySize; y++) {

            /*
            if (n == 0 && x ==10 && y==40) { //<= 11 && y <= 10) { // get(self->U, y,x, self->ySize) > 0) {
                //printf("n=%4d x=%2d y=%2d\n", n, x, y);
            }
            double Uorig = get(self->U, y,x, self->ySize);
            */
            double du = Diffusion2D(self->U, self->K, x,y, self->xSize, self->ySize, \
                    self->bPeriodic, self->gridSize);
            if (n == 1000 && x ==10 && y==40) { //<= 11 && y <= 10) { // get(self->U, y,x, self->ySize) > 0) {
                //printf("x=%2d y=%2d\n", x, y);
                //printf("%f += %f\n", self->uTotal, get(self->U, y,x, self->ySize));
            }
            self->uTotal += get(self->U, y,x, self->xSize, self->ySize);

            //-- timestep
            *idx(self->U, y,x, self->xSize, self->ySize) +=  du * self->dt;

            //-- Sources and Sinks
            *idx(self->U, y,x, self->xSize, self->ySize) += get(self->Source, y,x, self->xSize, self->ySize) * self->dt;
            *idx(self->U, y,x, self->xSize, self->ySize) -= get(self->Sink, y,x, self->xSize, self->ySize) * self->dt;


            if (get(self->U, y,x, self->xSize, self->ySize) > self->uMax)
                self->uMax = get(self->U, y,x, self->xSize, self->ySize);
            if (get(self->U, y,x, self->xSize, self->ySize) < self->uMin)
                self->uMin = get(self->U, y,x, self->xSize, self->ySize);
            /*
            double Unow = get(self->U, y,x, self->xSize, self->ySize);
            if (n <= 1000 && Unow != 0) {
                printf("n=%4d x=%2d y=%2d ", n, x, y);
                printf("du=%2.15f uTotal=%2.8f Uorig=%2.8f Unow=%2.15f\n", du, self->uTotal, Uorig, Unow);
            }
            */
        }
    }

    self->time = self->dt * (float)n;
    //sz = "%08d %12.11f %12.11f %4.1f %4.1f\n" % (n, self->time, self->uTotal, self->uMax, self->uMin)
    //print (sz)
    //AppendLog(sz)
    //AppendLog(n, self->time, self->uTotal, self->uMax, self->uMin)
}

void run(tSim *self)
{
    for (int n=0; n<self->nsteps; n++) {

        GetHeat(self, n * self->paintsteps);
        set(self->observations, n, 0, self->nsteps, 5, n);
        set(self->observations, n, 1, self->nsteps, 5, self->time);
        set(self->observations, n, 2, self->nsteps, 5, self->uTotal);
        set(self->observations, n, 3, self->nsteps, 5, self->uMax);
        set(self->observations, n, 4, self->nsteps, 5, self->uMin);
    }
}

int main(int argc, char **argv)
{
    tSim sim;

    __init__(&sim);

    run(&sim);

    /* report result */
    tSim *self = &sim;
    for (int i=0; i<self->nsteps; i++) {
        double n = get(sim.observations, i, 0, self->nsteps, 5);
        double time = get(sim.observations, i, 1, self->nsteps, 5);
        double uTotal = get(sim.observations, i, 2, self->nsteps, 5);
        double uMax = get(sim.observations, i, 3, self->nsteps, 5);
        double uMin = get(sim.observations, i, 4, self->nsteps, 5);
        //printf("%08.0f %12.11f %12.11f %4.1f %4.1f\n", n, time, uTotal, uMax, uMin);
        printf("%016.8f %12.11f %12.11f %12.8f %12.8f\n", n, time, uTotal, uMax, uMin);
    }

    return 0;
}

/*
    //--------------------------
    def SetSizes(self, szx, szy, p=False):
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


    //--------------------------
*/
