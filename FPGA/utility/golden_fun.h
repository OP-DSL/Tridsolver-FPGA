#include <sys/time.h>


#ifndef __GOLDEN_FUN_H__
#define __GOLDEN_FUN_H__







#if FPPREC == 0
#  define FP float
#elif FPPREC == 1
#  define FP double
#else
#  error "Macro definition FPPREC unrecognized for CUDA"
#endif

#define N_MAX 512
#define BATCH_SIZE 256


extern char *optarg;
extern int  optind, opterr, optopt;
static struct option options[] = {
  {"nx",   required_argument, 0,  0   },
  {"ny",   required_argument, 0,  0   },
  {"nz",   required_argument, 0,  0   },
  {"iter", required_argument, 0,  0   },
  {"opt",  required_argument, 0,  0   },
  {"prof", required_argument, 0,  0   },
  {"help", no_argument,       0,  'h' },
  {0,      0,                 0,  0   }
};



inline double elapsed_time(double *et) {
  struct timeval t;

  double old_time = *et;

  gettimeofday( &t, (struct timezone *)0 );
  *et = t.tv_sec + t.tv_usec*1.0e-6;

  return *et - old_time;
}


template <class DType>
class golden
{
public:
  golden(){};
  ~golden(){};
  void thomas_golden(DType* __restrict a, DType* __restrict b, DType* __restrict c,
      DType* __restrict d, DType* __restrict u, int N, int stride);

  double square_error(DType* golden, DType* FPGA, int nx, int ny, int nz);

};

#endif
