CFLAGS:=-Wall -O0
#CC=gcc
CC=clang

%.ll: %.c
	rm -f $@.tmp
	${CC} ${CFLAGS} -O0 -emit-llvm -S $<
	echo "\n\n;;; O0" >> $@.tmp
	cat $@ >> $@.tmp
	${CC} ${CFLAGS} -O1 -emit-llvm -S $<
	echo "\n\n;;; O1" >> $@.tmp
	cat $@ >> $@.tmp
	${CC} ${CFLAGS} -O2 -emit-llvm -S $<
	echo "\n\n;;; O2" >> $@.tmp
	cat $@ >> $@.tmp
	#${CC} -cc1 ${CFLAGS} -O0 -g -emit-llvm $<
	#echo "\n\n;;; O0 -g" >> $@.tmp
	#cat $@ >> $@.tmp
	mv $@.tmp $@

%.llc.cpp: %.ll
	llc -march=cpp -o $@ $<

ctest: ctest.o
ctest.o: ctest.c
ctest.ll: ctest.c

fib: fib.o
fib.o: fib.c
