#include <stdlib.h>

typedef struct {
    int x;
    int y;
} tCoord;

int main(int argc, char ** argv) {
    tCoord c;
    c.x = 1;
    c.y += 2;
    return 0;
}
