SHELL = /bin/bash
CC = gcc
CXX = nvcc
DEBUGFLAGS =
FLAGS =  -I/opt/cuda/include -L/opt/cuda/lib64
CFLAGS =  --std=c++11
CUDAFLAGS = -Xcompiler -fPIC -Xcompiler -static -arch=sm_61 
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
CP= cp -f
CMD= bash
BUILDSCRIPT= build.sh
.PHONY: clean
TARGETDIR = bin/
SOURCEDIR = lib/
BUILDDIR = build/
OBJDIR = objects/
MAKEDIRS = mkdir -p 
INSTALLDIR = src/tensorplus/

all: $(TARGETDIR)$(TARGET) $(TARGETDIR)$(TARGETMAC) $(TARGETDIR)$(TARGETWIN)

$(TARGETDIR)$(TARGETMAC): $(BUILDDIR)$(OBJDIR)$(OBJECTS) $(BUILDDIR)$(OBJDIR)$(CUOBJS)
	$(CC) $(CFLAGS) $(DEBUGFLAGS) $(LDFLAGS) $(GCCFLAGS) -o $(TARGETDIR)$(TARGETMAC) $(BUILDDIR)$(OBJDIR)$(OBJECTS) $(BUILDDIR)$(OBJDIR)$(CUOBJS) $(FLAGS) $(EXTRA_CUDA_FLAGS)

$(TARGETDIR)$(TARGETWIN): $(BUILDDIR)$(OBJDIR)$(OBJECTS) $(BUILDDIR)$(OBJDIR)$(CUOBJS)
	$(CC) $(CFLAGS) $(DEBUGFLAGS) $(LDFLAGS) $(GCCFLAGS)   -o $(TARGETDIR)$(TARGETWIN) $(BUILDDIR)$(OBJDIR)$(OBJECTS) $(BUILDDIR)$(OBJDIR)$(CUOBJS) $(FLAGS) $(EXTRA_CUDA_FLAGS)

$(TARGETDIR)$(TARGET): $(BUILDDIR)$(OBJDIR)$(OBJECTS) $(BUILDDIR)$(OBJDIR)$(CUOBJS)
	$(CC) $(CFLAGS) $(DEBUGFLAGS) $(LDFLAGS) $(GCCFLAGS) -o $(TARGETDIR)$(TARGET) $(BUILDDIR)$(OBJDIR)$(OBJECTS) $(BUILDDIR)$(OBJDIR)$(CUOBJS) $(FLAGS) $(EXTRA_CUDA_FLAGS)

$(BUILDDIR)$(OBJDIR)$(CUOBJS): $(SOURCEDIR)$(CUSOURCES)
	$(CXX) $(CFLAGS) $(DEBUGFLAGS) $(LDFLAGS) $(GCCFLAGS) $(OBJTARFLAGS) $(CUDAFLAGS) -c $(SOURCEDIR)$(CUSOURCES) $(FLAGS) $(EXTRA_CUDA_FLAGS)

$(BUILDDIR)$(OBJDIR)$(OBJECTS): $(SOURCEDIR)$(SOURCES)
	$(CC) $(CFLAGS) $(DEBUGFLAGS) $(LDFLAGS) $(GCCFLAGS) $(OBJTARFLAGS) -c $(SOURCEDIR)$(SOURCES) $(FLAGS)

clean:
	$(RM) $(BUILDDIR)$(OBJDIR)$(OBJECTS) $(BUILDDIR)$(OBJDIR)$(CUOBJS) $(TARGETDIR)$(TARGET) $(TARGETDIR)$(TARGETMAC) $(TARGETDIR)$(TARGETWIN)

install:
	#$(CP) $(TARGETDIR)$(TARGET) $(INSTALLDIR)$(TARGET)
	#$(CP) $(TARGETDIR)$(TARGETMAC) $(INSTALLDIR)$(TARGETMAC)
	#$(CP) $(TARGETDIR)$(TARGETWIN) $(INSTALLDIR)$(TARGETWIN)
	#$(CMD) $(BUILDSCRIPT)
