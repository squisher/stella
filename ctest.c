//#include <stdlib.h>
//#include <stdio.h>

typedef struct {
    int x;
    long long y;
} tCoord;

void setx(tCoord *t, int x) {
    t->x = x;
}

tCoord returnStruct() {
    tCoord tc;
    tc.x = 1;
    tc.y = 16384;
    return tc;
}

int main(int argc, char ** argv) {
    /*
    tCoord c[2];
    c[0] = returnStruct();
    */
    /*
    setx(&(c[1]), 1);
    c[0].x = 2;
    */
    //return c[0].x;
    tCoord c = returnStruct();
    return c.x;
}
