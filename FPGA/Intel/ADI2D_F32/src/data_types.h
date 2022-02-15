#include <CL/sycl.hpp>
#include <vector>
#include <iostream>
#include "dpc_common.hpp"
#if FPGA || FPGA_EMULATOR
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <CL/sycl/INTEL/ac_types/ac_int.hpp>
#endif


#ifndef __DATA_TYPES_H__
#define __DATA_TYPES_H__

const int unroll_factor = 2;
const int v_factor = 16;

using namespace sycl;

struct dPath {
  [[intel::fpga_register]] float data[8];
};


struct dPath16 {
  [[intel::fpga_register]] float data[16];
};

// Vector type and data size for this example.
size_t vector_size = 10000;
typedef std::vector<struct dPath16> IntVector; 
typedef std::vector<float> IntVectorS; 

using rd_pipe = INTEL::pipe<class pVec16_r, dPath16, 512000>;
using wr_pipe = INTEL::pipe<class pVec16_w, dPath16, 512000>;

#define UFACTOR 1

struct pipeS{
  pipeS() = delete;
  template <size_t idx>  struct struct_id;

  template <size_t idx>
  struct Pipes{
    using pipeA = INTEL::pipe<struct_id<idx>, dPath, 8000000>;
  };

  template <size_t idx>
  using PipeAt = typename Pipes<idx>::pipeA;
};

struct pipeM{
  pipeM() = delete;
  template <size_t idx>  struct struct_id;

  template <size_t idx>
  struct PipeM{
    using pipeA = INTEL::pipe<struct_id<idx>, dPath16, 8000000>;
  };

  template <size_t idx>
  using PipeAt = typename PipeM<idx>::pipeA;
};


struct pipeB{
  pipeB() = delete;
  template <size_t idx>  struct struct_id;

  template <size_t idx>
  struct PipeB{
    using pipeA = INTEL::pipe<struct_id<idx>, dPath16, 8000000>;
  };

  template <size_t idx>
  using PipeAt = typename PipeB<idx>::pipeA;
};


using PipeBlock = pipeS;

// using rd_pipe1 = INTEL::pipe<class rd_pipe1, dPath, 8>;
// using wr_pipe1 = INTEL::pipe<class wr_pipe1, dPath, 8>;


// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

struct data_G{
	unsigned short sizex;
	unsigned short sizey;
	unsigned short sizez;
	unsigned short xdim0;
	unsigned short end_index;
	unsigned short end_row;
	unsigned int gridsize;
    unsigned int total_itr_512;
    unsigned int total_itr_256;
	unsigned short outer_loop_limit;
	unsigned short endrow_plus2;
	unsigned short endrow_plus1;
	unsigned short endrow_minus1;
	unsigned short endindex_minus1;
};

struct data_G_3d{
	unsigned short sizex;
	unsigned short sizey;
	unsigned short sizez;
	unsigned short xdim;
	unsigned short xblocks;
	unsigned short grid_sizey;
	unsigned short grid_sizez;
	unsigned short limit_z;
	unsigned short offset_x;
	unsigned short tile_x;
	unsigned short offset_y;
	unsigned short tile_y;
	unsigned int plane_size;
	unsigned int gridsize_pr;
	unsigned int gridsize_da;
	unsigned int plane_diff;
	unsigned int line_diff;
	unsigned short outer_loop_limit;
	unsigned int total_itr;
	bool last_half;
	unsigned short batches;
};


// Trip count
const int max_size_y = 256;
const int min_size_y = 32;
const int avg_size_y = 256;

const int batch_s = 100;
const int max_block_x = (256/8 + 1);
const int min_block_x = (32/8 + 1);
const int avg_block_x = (256/8 + 1);

const int max_grid = max_block_x * max_size_y * max_size_y * batch_s;
const int min_grid = min_block_x * min_size_y * min_size_y * batch_s;
const int avg_grid = avg_block_x * avg_size_y * avg_size_y * batch_s;

const int max_grid_2 = (max_block_x * max_size_y * max_size_y)/2 * batch_s;
const int min_grid_2 = (min_block_x * min_size_y * min_size_y)/2 * batch_s;
const int avg_grid_2 = (avg_block_x * avg_size_y * avg_size_y)/2 * batch_s;

#endif
