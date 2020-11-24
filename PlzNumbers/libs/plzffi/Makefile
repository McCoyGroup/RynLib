CXX=g++

SRC=$(addprefix ./, FFIModule.cpp FFIParameters.cpp)
OBJ=$(addprefix ./, FFIModule.o FFIParameters.o)

EXE=libplzffi.so

FFLAGS= -O2 -fpic

all:
	$(CXX) -fpic -I".." -I$(PYTHON_INCLUDES) -I$(NUMPY_INCLUDES) -c FFIModule.cpp -o FFIModule.o
	$(CXX) -fpic -I".." -I$(PYTHON_INCLUDES) -I$(NUMPY_INCLUDES) -c FFIParameters.cpp -o FFIParameters.o
	$(CXX) -shared -fpic $(FFLAGS) $(OBJ) $(INCLUDE_PATH) $(LIB_PATH) $(LIBSHDF) $(LIBS) -o $(EXE)
	cp $(EXE) ../$(EXE)

clean:
	rm -rf $(OBJ) *.o *.mod
