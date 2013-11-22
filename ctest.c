#include <stdlib.h>

char and(int a, int b, int c) {
  if (a && b && c)
    return 'y';
  else
    return 'n';
}

int main(int argc, char ** argv) {
  exit(and(1,1,0) == 'y');
}
