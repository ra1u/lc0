
OFILE=wino_perf
CXXFLAGS=--std=c++14 -O3 -march=native
WINO_ISPCPATH=./ispc/objs
ISPC_O=${WINO_ISPCPATH}/winograd_ispc.o

${OFILE}: wino_perf.cc
	$(CXX)  ${CXXFLAGS}  -o ${OFILE} -I${WINO_ISPCPATH}  ${OFILE}.cc ${ISPC_O}

ispc:
	$(MAKE) -C ./ispc/ ispc


clean:
	rm -f ${OFILE}

.PHONY: ispc
