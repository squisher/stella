#include <stdlib.h>

char and(int a, int b) {
  if (a && b)
    return 'y';
  else
    return 'n';
}

int main(int argc, char ** argv) {
  exit(and(1,2) == 'y');
}
