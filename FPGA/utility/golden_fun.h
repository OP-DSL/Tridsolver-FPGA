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


void print_help() ;
void timing_start(int prof, double *timer);
void timing_end(int prof, double *timer, double *elapsed_accumulate, char *str);
void thomas_golden(float* __restrict a, float* __restrict b, float* __restrict c, float* __restrict d, float* __restrict u, int N, int batch);
double square_error(float* golden, float* FPGA, int nx, int ny, int nz);

#endif
