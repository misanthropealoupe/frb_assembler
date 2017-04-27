include makefile.local
SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .cpp .hpp .o .so

all: assembler.hpp
	g++ -ggdb -std=c++11 -fpermissive -pthread -I$(INC) -I$(INCDIR2) -I$(NPINCDIR) -I$(PYTHON_INC) -L$(LIB) assembler.cpp -o assembler -lfftw3 -lrf_pipelines -lbonsai -lm -ljsoncpp -lpng -lmpl_interface -ldedisp-contain -lyaml-cpp
