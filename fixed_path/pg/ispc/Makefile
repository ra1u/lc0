
ISPC=ispc
OBJS_DIR=objs
INFILE=winograd
CXXFLAGS=--std=c++14 -O3 -march=native


export PATH=/home/ralu/misc/software/category/compilers/ispc:$(shell printenv PATH)

ispc: dirs
	$(ISPC) -O2 --arch=x86-64 --target=host ${INFILE}.ispc -o ${OBJS_DIR}/${INFILE}_ispc.o -h ${OBJS_DIR}/${INFILE}_ispc.h

ispc_cpp: dirs
	$(ISPC) -O2 --arch=x86-64 --target=generic-32 ${INFILE}.ispc  --emit-c++ -o ${OBJS_DIR}/${INFILE}_ispc.cpp -h ${OBJS_DIR}/${INFILE}_ispc.h
	g++  ${CXXFLAGS} -o ${OBJS_DIR}/${INFILE}_ispc.o -c ${OBJS_DIR}/${INFILE}_ispc.cpp




dirs: ${OBJS_DIR}

${OBJS_DIR}:
	mkdir -p ${OBJS_DIR} 

clean:
	rm -f ${OBJS_DIR}/*
	rmdir ${OBJS_DIR} || true 


.PHONY: clean dirs ispc
