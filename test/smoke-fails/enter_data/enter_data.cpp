#include <hip_runtime.h>
#include <omp.h>
#include <stdio.h>
#include <unistd.h>

#define PAGE_SIZE sysconf(_SC_PAGESIZE)
#define GB (1024 * 1024 * 1024)
const size_t SIZE = .5 * 600 * 1024L * 1024L;

#pragma omp requires unified_shared_memory

int main() {
  printf("Total Memory = %.6lf GB\n", (double)SIZE * sizeof(double) / GB);
  printf("Using alignment value of %ld bytes \n", PAGE_SIZE);
  // Initialize device runtime and state
  int dummy = 0;
#pragma omp target
  dummy += 1;
  double *p = NULL;
  hipError_t e = hipHostMalloc((void **)&p, SIZE * sizeof(double));
  if (e != hipSuccess) {
    e = hipGetLastError();
    printf("hipHostMalloc returned %d:%s\n ", e, hipGetErrorString(e));
    abort();
  }
  printf("======> hipHostMalloc returned p:%p\n", (void *)p);
  double mem_size = (double)SIZE * sizeof(double) / GB;
  for (int j = 0; j < 4; ++j) {
    double start = omp_get_wtime();
#pragma omp target enter data map(always, to : p[:SIZE])
    double end = omp_get_wtime();
    printf("data MAP (to:) time = %.6lf Sec.  BW = %.6lf GB/S\n", end - start,
           mem_size / (end - start));
  }
  e = hipHostFree(p);
  return 0;
}