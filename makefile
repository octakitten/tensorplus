SHELL = /bin/sh
CC = nvcc
CXX = nvcc
LINK.cc = nvcc
DEBUGFLAGS = --debug
FLAGS = -std=c++11 -Iinclude 
CFLAGS = -forward-unknown-to-host-compiler -fPIC
RELEASEFLAGS = -02 -D NDEBUG
LDFLAGS = -shared
TARGETFLAGS = --output-directory $(BUILDDIR)
OBJTARFLAGS = --output-directory $(BUILDDIR)$(OBJDIR)
GCCFLAGS = -allow-unsupported-compiler

SOURCES += tensorplus.c
CUSOURCES += tensor.cu
HEADERS += defines.h includes.h
OBJECTS = tensorplus.o
CUOBJS = tensor.o
TARGET = tensorplus.so

RM= rm -f
.PHONY: all clean
SOURCEDIR = src/
BUILDDIR = build/
OBJDIR = objects/

$(BUILDDIR)$(TARGET): $(BUILDDIR)$(OBJDIR)$(OBJECTS) $(BUILDDIR)$(OBJDIR)$(CUOBJS)
	$(CC) $(FLAGS) $(DEBUGFLAGS) $(LDFLAGS) $(TARGETFLAGS) $(GCCFLAGS) -o $(BUILDDIR)$(TARGET) $(BUILDDIR)$(OBJDIR)$(OBJECTS) $(BUILDDIR)$(OBJDIR)$(CUOBJS)

$(BUILDDIR)$(OBJDIR)$(OBJECTS): $(SOURCEDIR)$(SOURCES)
	$(CC) $(FLAGS) $(CFLAGS) $(DEBUGFLAGS) $(LDFLAGS) $(GCCFLAGS) $(OBJTARFLAGS) -c $(SOURCEDIR)$(SOURCES)

$(BUILDDIR)$(OBJDIR)$(CUOBJS): $(SOURCEDIR)$(CUSOURCES)
	$(CC) $(FLAGS) $(CFLAGS) $(DEBUGFLAGS) $(LDFLAGS) $(GCCFLAGS) $(OBJTARFLAGS) -c $(SOURCEDIR)$(CUSOURCES)

clean:
	$(RM) $(BUILDDIR)$(OBJDIR)$(OBJECTS) $(BUILDDIR)$(TARGET)