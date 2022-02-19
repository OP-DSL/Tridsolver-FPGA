#include "data_types.h"
#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <CL/sycl/INTEL/ac_types/ac_int.hpp>

#ifndef __BTHOMAS_H__
#define __BTHOMAS_H__

#ifdef __SYCL_DEVICE_ONLY__
  #define CL_CONSTANT __attribute__((opencl_constant))
#else
  #define CL_CONSTANT
#endif

#define PRINTF(format, ...) { \
            static const CL_CONSTANT char _format[] = format; \
            sycl::ext::oneapi::experimental::printf(_format, ## __VA_ARGS__); }




template <size_t idx>  struct thomas_interleave_id;
template<bool FPPREC, class DType, int DMAX, int Pidx1, int Pidx2, int Pidx3>
void thomas_interleave(queue &q, ac_int<14,true> d0, ac_int<15,true> B, int ReadLimit,  ac_int<12,true> n_iter){

	event e = q.submit([&](handler &h) {
    h.single_task<class thomas_interleave_id<Pidx1>>([=] () [[intel::kernel_args_restrict]]{

		#define SHIFT (3-FPPREC)
		#define VFACTOR ((1 << SHIFT))
		#define DSIZE 	(256/VFACTOR)
		#define NBLK 	(9 <<FPPREC)

    	[[intel::disable_loop_pipelining]]
    	for(unsigned short u_itr = 0; u_itr < n_iter; u_itr++){

			const int n_blk = NBLK;


			// [[intel::force_pow2_depth(0)]] 
			struct dPath  d2[DMAX*NBLK*2];

			ac_int<17,true> batd1 = 0;
			ac_int<8,true> id1 =0;
			ac_int<13,true> jd1 = 0;
			ac_int<17,true> Bp1 = B+1;

			int total_itr = B*NBLK*d0 + NBLK*d0;

			struct dPath rand_vec;
			pipeS::PipeAt<Pidx3>::write(rand_vec);

			[[intel::initiation_interval(1)]]
			for(int itr= 0; itr < total_itr; itr++){

					ac_int<17,true> bat = batd1;
					ac_int<8,true> i = id1;
					ac_int<13,true> j = jd1;

					if(j == d0 -1){
						jd1 = 0;
					} else {
						jd1++;
					}

					if(j == d0 -1 && i == NBLK -1){
						id1 = 0;
						batd1++;
					} else if(j == d0 -1){
						id1++;
					}

					unsigned int offsetR = ((bat & 1) == 0) ?  DMAX*NBLK : 0;
					unsigned int offsetW = ((bat & 1) == 0) ?  0 : DMAX*NBLK;

					ac_int<21,true> countr1 = (bat*NBLK) + i;
					int count = countr1 * d0 + j;
					struct dPath  tmp_d;
					if(count < ReadLimit){
						tmp_d = pipeS::PipeAt<Pidx1>::read();
					} 
					int indW = j*NBLK + i + offsetW;
					d2[indW] = tmp_d;

					int indR = i*d0+j + offsetR;
					struct dPath  tmp_R = d2[indR];
					if(bat > 0){
						pipeS::PipeAt<Pidx2>::write(tmp_R);
					}
				}
			}

		#undef SHIFT
		#undef VFACTOR
		#undef DSIZE
		#undef NBLK

	});
    });

    // return e;
}

// template <size_t idx>  struct controlSig_generator_id;
// template <int Pidx1>
// event controlSig_generator(queue &q, ac_int<12,true> n_iter){
// 	event e = q.submit([&](handler &h) {
//     h.single_task<class controlSig_generator_id<Pidx1>>([=] () [[intel::kernel_args_restrict]]{
// 		[[intel::initiation_interval(1)]]
// 		for(int i = 0; i < n_iter; i++){
// 			struct dPath rand_vec;
// 			pipeS::PipeAt<Pidx1>::write(rand_vec);
// 		}
// 	});
//     });

//     return e;

// }



template <size_t idx>  struct thomas_generate_r_id;
template<bool FPPREC, class DType, int DMAX, int Pidx1, int Pidx2>
void thomas_generate_r(queue &q, ac_int<14,true> d0, ac_int<15,true> B, int count_limit, ac_int<12,true> n_iter){


	event e = q.submit([&](handler &h) {
    h.single_task<class thomas_generate_r_id<Pidx1>>([=] () [[intel::kernel_args_restrict]]{
		#define SHIFT (3-FPPREC)
		#define VFACTOR ((1 << SHIFT))
		#define DSIZE 	(256/VFACTOR)
		#define NBLK 	(45<<FPPREC)
		#define NBLK_W 	(9<<FPPREC)




    	[[intel::disable_loop_pipelining]]
		for(unsigned short u_itr = 0; u_itr < n_iter; u_itr++){

			const int n_blk = NBLK;
			
			// [[intel::force_pow2_depth(0)]] 
			struct dPath  r_mem[DMAX*NBLK*2];


			ac_int<17,true> batd2 = 0;
			ac_int<13,true> id2 =0;
			ac_int<8,true>  kd2 = 0;

			ac_int<17,true> Bp1 = B+1;
			int total_itr = B*NBLK*d0 + NBLK*d0;

			struct dPath window_b2[NBLK], window_c2[NBLK];
			pipeS::PipeAt<Pidx2>::read();

			[[intel::ivdep(NBLK)]]
			[[intel::initiation_interval(1)]]
			for(int itr= 0; itr < total_itr; itr++){

				ac_int<17,true> bat = batd2;
				ac_int<13,true> i = id2;
				ac_int<8,true> k = kd2;

				if(k == NBLK -1){
					kd2 = 0;
				} else {
					kd2++;
				}

				if(k == NBLK -1 && i == d0 -1){
					id2 = 0;
					batd2++;
				} else if(k == NBLK -1){
					id2++;
				}


				struct dPath vec_bb_r = window_b2[k];
				struct dPath vec_cc_r = window_c2[k];

				struct dPath r_vec;
				struct dPath vec_bb_w, vec_cc_w;

				#pragma unroll
				for(int v =0; v < VFACTOR; v++){
					DType aa_read = (i == 0 || i == d0 -1) ? 0.0 : -0.5; 
					DType bb_read = (i == 0 || i == d0 -1) ? 1.0 :  2.0; 
					DType cc_read = (i == 0 || i == d0 -1) ? 0.0 : -0.5;  

					DType bb_old = vec_bb_r.data[v]; 
					DType cc_old = vec_cc_r.data[v]; 



					DType denom = (i == 0) ? bb_read : (bb_read - aa_read*cc_old);
					DType r = 1.0/denom; //r_read.data[v]; 
					DType c_w1 = cc_read;

					DType b_wr = 1.0;
					DType c_wr = c_w1*r;

					r_vec.data[v] = r;

					vec_cc_w.data[v] = c_wr;


				}


				window_b2[k] = vec_bb_w;
				window_c2[k] = vec_cc_w;

				unsigned int offsetR = ((bat & 1) == 0) ?  DMAX*NBLK : 0;
				unsigned int offsetW = ((bat & 1) == 0) ?  0 : DMAX*NBLK;

				int indW =  k*d0+i+offsetW;
				r_mem[indW] = r_vec;
	

				ac_int<16,true> count = i*NBLK+k;
				ac_int<16,true> blk = count % (NBLK_W*d0);
				ac_int<4,true> blk_id = count / (NBLK_W*d0);

				ac_int<13,true> i_new = blk / NBLK_W;
				ac_int<8,true> k_new = (blk%NBLK_W) + blk_id*NBLK_W;

				int indR =  k_new*d0+i_new + offsetR;

				struct dPath r_vecR = r_mem[indR];

				if(bat > 0 && itr < count_limit+ NBLK*d0){
					pipeS::PipeAt<Pidx1>::write(r_vecR);
				}

			}
		}

		#undef SHIFT
		#undef VFACTOR
		#undef DSIZE
		#undef NBLK
		#undef NBLK_W
	
	});
    });

    // return e;
}





template <size_t idx>  struct thomas_forward_id;
template<bool FPPREC, class DType, int DMAX, int Pidx1, int Pidx2>
void thomas_forward(queue &q, ac_int<14,true> d0, ac_int<15,true> B, ac_int<12,true> n_iter){


	event e = q.submit([&](handler &h) {
    h.single_task<class thomas_forward_id<Pidx1>>([=] () [[intel::kernel_args_restrict]]{
		#define SHIFT (3-FPPREC)
		#define VFACTOR ((1 << SHIFT))
		#define DSIZE 	(256/VFACTOR)
		#define NBLK 	(9<<FPPREC)

    	[[intel::disable_loop_pipelining]]
		for(unsigned short u_itr = 0; u_itr < n_iter; u_itr++){

			const int n_blk = NBLK;

			// [[intel::force_pow2_depth(0)]] 
			struct dPath  c2_fw[DMAX*NBLK*2];
			// [[intel::force_pow2_depth(0)]] 
			struct dPath  d2_fw[DMAX*NBLK*2];


			ac_int<17,true> batd2 = 0;
			ac_int<13,true> id2 =0;
			ac_int<8,true>  kd2 = 0;

			ac_int<17,true> Bp1 = B+1;
			int total_itr = B*NBLK*d0 + NBLK*d0;

			struct dPath window_b2[NBLK], window_c2[NBLK], window_d2[NBLK];

			[[intel::ivdep(NBLK)]]
			[[intel::initiation_interval(1)]]
			for(int itr= 0; itr < total_itr; itr++){

				ac_int<17,true> bat = batd2;
				ac_int<13,true> i = id2;
				ac_int<8,true> k = kd2;

				if(k == NBLK -1){
					kd2 = 0;
				} else {
					kd2++;
				}

				if(k == NBLK -1 && i == d0 -1){
					id2 = 0;
					batd2++;
				} else if(k == NBLK -1){
					id2++;
				}


				struct dPath d2_read, r_read;
				if(bat < B){
					d2_read = pipeS::PipeAt<Pidx1>::read();
					r_read = pipeS::PipeAt<Pidx1+1>::read();
				}

				struct dPath vec_bb_r = window_b2[k];
				struct dPath vec_dd_r = window_d2[k];
				struct dPath vec_cc_r = window_c2[k];

				struct dPath b2_fw_write, d2_fw_write;
				struct dPath vec_bb_w, vec_dd_w, vec_cc_w;

				#pragma unroll
				for(int v =0; v < VFACTOR; v++){
					DType aa_read = (i == 0 || i == d0 -1) ? 0.0 : -0.5; 
					DType bb_read = (i == 0 || i == d0 -1) ? 1.0 :  2.0; 
					DType cc_read = (i == 0 || i == d0 -1) ? 0.0 : -0.5; 
					DType dd_read = d2_read.data[v]; 

					DType bb_old = vec_bb_r.data[v]; 
					DType dd_old = vec_dd_r.data[v]; 
					DType cc_old = vec_cc_r.data[v]; 



					DType denom = (i == 0) ? bb_read : (bb_read - aa_read*cc_old);
					DType r = r_read.data[v]; 
					DType c_w1 = cc_read;
					DType d_w1 = (i == 0) ? dd_read : (dd_read - aa_read*dd_old);

					DType b_wr = 1.0;
					DType c_wr = c_w1*r;
					DType d_wr = d_w1*r;




					b2_fw_write.data[v] = b_wr;
					d2_fw_write.data[v] = d_wr;

					vec_bb_w.data[v] = b_wr;
					vec_dd_w.data[v] = d_wr;
					vec_cc_w.data[v] = c_wr;


					// if(bat < B){
					// 	PRINTF("(%f, %f) ", r, r_read.data[v]);
					// }

				}

				// if(bat < B && Pidx1 == 21){
				// 	PRINTF(" itr : %d\n", itr);
				// }

				window_b2[k] = vec_bb_w;
				window_d2[k] = vec_dd_w;
				window_c2[k] = vec_cc_w;

				unsigned int offsetR = ((bat & 1) == 0) ?  DMAX*NBLK : 0;
				unsigned int offsetW = ((bat & 1) == 0) ?  0 : DMAX*NBLK;

				int indW =  k*d0+i+offsetW;
				c2_fw[indW] = vec_cc_w;
				d2_fw[indW] = d2_fw_write;
				ac_int<13,true> i_rev = d0-i -1;
				int indR =  k*d0+ i_rev + offsetR;

				struct dPath c2_fw_stmR = c2_fw[indR];
				struct dPath d2_fw_stmR = d2_fw[indR];

				if(bat > 0){
					pipeS::PipeAt<Pidx2>::write(c2_fw_stmR);
					pipeS::PipeAt<Pidx2+1>::write(d2_fw_stmR);
				}

				ac_int<17,true> bat_1 = (bat-1);
				int count = (int)(bat_1 * NBLK * d0) + i*NBLK+k;

				// if(bat > 0){
				// 		// PRINTF("%d %f %f %f %f %f %f %f %f  \n", count , c2_fw_stmR.data[0], c2_fw_stmR.data[1], c2_fw_stmR.data[2], \
				// 		// c2_fw_stmR.data[3], c2_fw_stmR.data[4], c2_fw_stmR.data[5], c2_fw_stmR.data[6], c2_fw_stmR.data[7]);

				// 		PRINTF("%d %f %f %f %f %f %f %f %f  \n", count , d2_fw_stmR.data[0], d2_fw_stmR.data[1], d2_fw_stmR.data[2], \
				// 			d2_fw_stmR.data[3], d2_fw_stmR.data[4], d2_fw_stmR.data[5], d2_fw_stmR.data[6], d2_fw_stmR.data[7]);
				// }



			}
		}

		#undef SHIFT
		#undef VFACTOR
		#undef DSIZE
		#undef NBLK
	
	});
    });

    // return e;
}



template <size_t idx>  struct thomas_backward_id;
template<bool FPPREC, class DType, int DMAX, int Pidx1, int Pidx2>
void thomas_backward(queue &q, ac_int<14,true> d0, ac_int<15,true> B, int ReadLimit,   ac_int<12,true> n_iter){


	event e = q.submit([&](handler &h) {
    h.single_task<class thomas_backward_id<Pidx1>>([=] () [[intel::kernel_args_restrict]]{

		#define SHIFT (3-FPPREC)
		#define VFACTOR ((1 << SHIFT))
		#define DSIZE 	(256/VFACTOR)
		#define NBLK 	((9<<FPPREC))

   		[[intel::disable_loop_pipelining]]
    	for(unsigned short u_itr = 0; u_itr < n_iter; u_itr++){

			const int n_blk = NBLK;

			// [[intel::force_pow2_depth(0)]] 
			struct dPath  u2[DMAX*NBLK*2];
			struct dPath window_u2[NBLK];

			ac_int<17,true> batd3 = 0;
			ac_int<13,true> id3 =0;
			ac_int<8,true>  kd3 = 0;

			ac_int<17,true> Bp1 = B+1;
			int total_itr = B*NBLK*d0 + NBLK*d0;

			[[intel::ivdep(NBLK)]]
			[[intel::initiation_interval(1)]]
			for(int itr= 0; itr < total_itr; itr++){

				ac_int<17,true> bat = batd3;
				ac_int<13,true> id = id3;
				ac_int<8,true> k = kd3;

				if(k == NBLK -1){
					kd3 = 0;
				} else {
					kd3++;
				}

				if(k == NBLK -1 && id == d0 -1){
					id3 = 0;
					batd3++;
				} else if(k == NBLK -1){
					id3++;
				}

				ac_int<13,true> i = d0 -1 -id;

				struct dPath d2_fw_read;
				struct dPath c2_fw_read;

				if(bat < B){
					c2_fw_read = pipeS::PipeAt<Pidx1>::read();
					d2_fw_read = pipeS::PipeAt<Pidx1+1>::read();
				}
				struct dPath u2_write;

				struct dPath vec_u2_r = window_u2[k];
				struct dPath vec_u2_w;

				ac_int<17,true> bat_1 = (bat-1);
				int count = (int)(bat_1 * NBLK * d0) + id*NBLK+k;

				#pragma unroll
				for(int v = 0; v < VFACTOR; v++){
					DType dd_read = d2_fw_read.data[v];
					DType cc_read = c2_fw_read.data[v];

					DType u_pre = vec_u2_r.data[v];
					DType numer_l = dd_read;
					DType numer_o = (dd_read - cc_read * u_pre);
					DType numer = (i == d0-1) ? numer_l : numer_o;

					DType u_new = numer;
					u2_write.data[v] = u_new;
					vec_u2_w.data[v] = u_new;

					// if(bat < B){
					// 	PRINTF("%f ", u_new);
					// }




				}

				// if(bat < B && Pidx1 == 23){
				// 	PRINTF("u_iter itr : %d %d\n", u_itr, itr);
				// }

				// PRINTF(" itr : %d\n", itr);

				unsigned int offsetR = ((bat & 1) == 0) ?  DMAX*NBLK : 0;
				unsigned int offsetW = ((bat & 1) == 0) ?  0 : DMAX*NBLK;

				int indW = k* d0 + i + offsetW;
				int indR = id*NBLK+k + offsetR;

				u2[indW] = u2_write;
				struct dPath u_stm_R = u2[indR];


				// PRINTF("%d %f %f %f %f %f %f %f %f  \n", count , u_stm_R.data[0], u_stm_R.data[1], u_stm_R.data[2], \
				// 	u_stm_R.data[3], u_stm_R.data[4], u_stm_R.data[5], u_stm_R.data[6], u_stm_R.data[7]);
				if(bat >0 && count < ReadLimit){
					pipeS::PipeAt<Pidx2>::write(u_stm_R);
				}
				window_u2[k] = vec_u2_w;
			}
		}

		#undef SHIFT
		#undef VFACTOR
		#undef DSIZE
		#undef NBLK

	});
    });

    // return e;

}

#endif
