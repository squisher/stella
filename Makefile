CFLAGS:=-Wall
#CC=gcc
CC=clang

%.s: %.c
	rm -f $@.tmp
	${CC} ${CFLAGS} -O0 -emit-llvm -S $<
	echo -e "\n\n;;; O0" >> $@.tmp
	cat $@ >> $@.tmp
	${CC} ${CFLAGS} -O1 -emit-llvm -S $<
	echo -e "\n\n;;; O1" >> $@.tmp
	cat $@ >> $@.tmp
	${CC} ${CFLAGS} -O2 -emit-llvm -S $<
	echo -e "\n\n;;; O2" >> $@.tmp
	cat $@ >> $@.tmp
	mv $@.tmp $@

ctest: ctest.o
ctest.o: ctest.c
ctest.s: ctest.c

fib: fib.o
fib.o: fib.c
