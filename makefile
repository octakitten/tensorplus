SHELL = /bin/bash
CC = gcc
CXX = nvcc
DEBUGFLAGS =
FLAGS =  -I/opt/cuda/include -L/opt/cuda/lib64
CFLAGS =  --std=c++11
CUDAFLAGS = -Xcompiler -fPIC -Xcompiler -static -arch=sm_61 $(EXTRA_CUDA_FLAGS)
RELEASEFLAGS =
LDFLAGS = -shared
TARGETFLAGS = #--output-directory $(TARGETDIR)
OBJTARFLAGS = #--output-directory $(BUILDDIR)$(OBJDIR)
GCCFLAGS =

SOURCES = tensorplus.c
CUSOURCES = tensor.cu
HEADERS = defines.h includes.h
OBJECTS = tensorplus.o
CUOBJS = tensor.o
TARGET = tensorplus.so
TARGETMAC = tensorplus.dylib
TARGETWIN = tensorplus.dll

RM= rm -f
.PHONY: clean
TARGETDIR = 
SOURCEDIR = lib/
BUILDDIR = #build/
OBJDIR = #objects/
MAKEDIRS = mkdir -p 
INSTALLDIR = $(out)

all: $(TARGETDIR)$(TARGET) $(TARGETDIR)$(TARGETMAC) $(TARGETDIR)$(TARGETWIN)

$(TARGETDIR)$(TARGETMAC): $(BUILDDIR)$(OBJDIR)$(OBJECTS) $(BUILDDIR)$(OBJDIR)$(CUOBJS)
	$(CC) $(FLAGS) $(CFLAGS) $(DEBUGFLAGS) $(LDFLAGS) $(GCCFLAGS) -o $(TARGETDIR)$(TARGETMAC) $(BUILDDIR)$(OBJDIR)$(OBJECTS) $(BUILDDIR)$(OBJDIR)$(CUOBJS)

$(TARGETDIR)$(TARGETWIN): $(BUILDDIR)$(OBJDIR)$(OBJECTS) $(BUILDDIR)$(OBJDIR)$(CUOBJS)
	$(CC) $(FLAGS) $(CFLAGS) $(DEBUGFLAGS) $(LDFLAGS) $(GCCFLAGS)   -o $(TARGETDIR)$(TARGETWIN) $(BUILDDIR)$(OBJDIR)$(OBJECTS) $(BUILDDIR)$(OBJDIR)$(CUOBJS)

$(TARGETDIR)$(TARGET): $(BUILDDIR)$(OBJDIR)$(OBJECTS) $(BUILDDIR)$(OBJDIR)$(CUOBJS)
	$(CC) $(FLAGS) $(CFLAGS) $(DEBUGFLAGS) $(LDFLAGS) $(GCCFLAGS) -o $(TARGETDIR)$(TARGET) $(BUILDDIR)$(OBJDIR)$(OBJECTS) $(BUILDDIR)$(OBJDIR)$(CUOBJS)

$(BUILDDIR)$(OBJDIR)$(CUOBJS): $(SOURCEDIR)$(CUSOURCES)
	$(CXX) $(FLAGS) $(CFLAGS) $(DEBUGFLAGS) $(LDFLAGS) $(GCCFLAGS) $(OBJTARFLAGS) $(CUDAFLAGS) -c $(SOURCEDIR)$(CUSOURCES)

$(BUILDDIR)$(OBJDIR)$(OBJECTS): $(SOURCEDIR)$(SOURCES)
	$(CC) $(FLAGS) $(CFLAGS) $(DEBUGFLAGS) $(LDFLAGS) $(GCCFLAGS) $(OBJTARFLAGS) -c $(SOURCEDIR)$(SOURCES)

clean:
	$(RM) $(BUILDDIR)$(OBJDIR)$(OBJECTS) $(BUILDDIR)$(OBJDIR)$(CUOBJS) $(TARGETDIR)$(TARGET) $(TARGETDIR)$(TARGETMAC) $(TARGETDIR)$(TARGETWIN)

install:
