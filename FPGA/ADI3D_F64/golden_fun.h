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
void thomas_golden(double* __restrict a, double* __restrict b, double* __restrict c, double* __restrict d, double* __restrict u, int N, int stride);
void trid_cpu(FP* __restrict a, FP* __restrict b, FP* __restrict c, FP* __restrict d, FP* __restrict u, int N, int stride) ;
void adi_cpu(FP lambda, FP* __restrict u, FP* __restrict du, FP* __restrict ax, FP* __restrict bx, FP* __restrict cx, FP* __restrict ay, FP* __restrict by, FP* __restrict cy, FP* __restrict az, FP* __restrict bz, FP* __restrict cz, int nx, int ny, int nz, double *elapsed_preproc, double *elapsed_trid_x, double *elapsed_trid_y, double *elapsed_trid_z, int prof);
double square_error(double* golden, double* FPGA, int size);

#endif
