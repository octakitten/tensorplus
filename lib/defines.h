// define global variables and preproc directives
// these should be included in all files
// 
// define cuda functionality first if available
#define __DEBUG__ 1

#ifndef __CUDACC__
#endif

#if __CUDACC__ > 0
#define CUDA_ENABLED 1
#else
#define CUDA_DISABLED 1
#endif

#if __CUDACC__ > 1
#define CUDA_MULTI 1
#else 
#define CUDA_MULTI 0
#endif

#define GPU 1
#define CPU 0