#include "data_types.h"
#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <CL/sycl/INTEL/ac_types/ac_int.hpp>

#ifndef __DPATH_H__
#define __DPATH_H__



// TDMA Modules


template <size_t idx>  struct interleaved_row_block8_id;
template <bool FPPREC, int DMAX, int Pidx1, int Pidx2>
void interleaved_row_block8(queue &q, ac_int<13,true> M, ac_int<13,true> N, ac_int<13,true> B, ac_int<13,true> n_iter, bool interleave){

	event e = q.submit([&](handler &h) {
    h.single_task<class interleaved_row_block8_id<Pidx1>>([=] () {
		#define SHIFT (3-FPPREC)
		#define VFACTOR ((1 << SHIFT))
		#define DSIZE 	(256/VFACTOR)
		#define ADJUST (VFACTOR-1)

    	[[intel::disable_loop_pipelining]]
		for(unsigned short u_itr = 0; u_itr < n_iter; u_itr++){

			ac_int<13,true> TileX, TileY;
			ac_int<25,true> NTiles;
			ac_int<9,true> XBlocks = (M >> SHIFT);
			unsigned int offset;

			const int N_CU = VFACTOR;
			switch(interleave){
				case true: {TileX=XBlocks; TileY=N_CU; NTiles = (B*N+ADJUST)>>SHIFT; break;}
				case false: {TileX=8; TileY=N; NTiles = (((XBlocks*B+ADJUST)>>SHIFT)); break;}
				default: {TileX=XBlocks; TileY=N_CU; NTiles = (B*N+ADJUST)>>SHIFT; break;}
			}


			// [[intel::force_pow2_depth(0)]] 
			struct dPath tmp_M[DMAX*N_CU*2];

			ac_int<25,true> NTilesp1 = NTiles+1;
			ac_int<25,true> id = 0;
			ac_int<13,true> jd =0, kd = 0;
			int total_itr = NTilesp1*TileX*TileY;

			[[intel::initiation_interval(1)]]
			for(int itr= 0; itr < total_itr; itr++){

				ac_int<25,true> i = id;
				ac_int<13,true> j = jd, k = kd;

				if(k == TileX -1){
					kd = 0;
				} else {
					kd++;
				}

				if(k == TileX -1 && j == TileY -1){
					jd = 0;
					id++;
				} else if(k == TileX -1){
					jd++;
				}
				unsigned int offsetR = ((i & 1) == 0) ?  DMAX*N_CU : 0;
				unsigned int offsetW = ((i & 1) == 0) ?  0 : DMAX*N_CU;

				bool cmpW = !interleave || (i*TileY + j < B*N);
				int indW = k*TileY+j + offsetW;
				struct dPath tmpW;
				if(cmpW && i < NTiles){
					tmpW = pipeS::PipeAt<Pidx1>::read();
				}
				tmp_M[indW] = tmpW;

				int indR = j*TileX + k + offsetR;
				struct dPath tmpR = tmp_M[indR];
				if(i > 0){
					pipeS::PipeAt<Pidx2>::write(tmpR);
				}
			}
		}
		#undef SHIFT
		#undef VFACTOR
		#undef DSIZE
		#undef ADJUST

	});
    });

    // return e;

}


template <size_t idx>  struct row2col_id;
template <bool FPPREC, int DMAX, int Pidx1, int Pidx2>
void row2col(queue &q, ac_int<13,true> M, ac_int<13,true> N, ac_int<15,true> B,  ac_int<13,true> n_iter){


	event e = q.submit([&](handler &h) {
    h.single_task<class row2col_id<Pidx1>>([=] () {
		#define SHIFT (3-FPPREC)
		#define VFACTOR ((1 << SHIFT))
		#define DSIZE 	(256/VFACTOR)
		#define ADJUST (VFACTOR-1)

    	[[intel::disable_loop_pipelining]]
		for(unsigned short u_itr = 0; u_itr < n_iter; u_itr++){


			ac_int<13,true> TileX, TileY;
			ac_int<21,true> NTiles;
			ac_int<9,true> XBlocks = (M >> SHIFT);
			unsigned int offset;

			TileX = XBlocks;
			TileY = N;
			NTiles = B;

			// [[intel::force_pow2_depth(0)]] 
			struct dPath tmp_M[DMAX*DMAX/VFACTOR*2];
			struct dPath tmp;

			ac_int<19,true> NTilesp1 = B+1;
			ac_int<19,true> id = 0;
			ac_int<13,true> jd =0, kd = 0;
			int total_itr = NTilesp1*TileX*TileY;

			[[intel::initiation_interval(1)]]
			for(int itr= 0; itr < total_itr; itr++){

				ac_int<19,true> i = id;
				ac_int<13,true> j = jd, k = kd;

				if(k == TileX -1){
					kd = 0;
				} else {
					kd++;
				}

				if(k == TileX -1 && j == TileY -1){
					jd = 0;
					id++;
				} else if(k == TileX -1){
					jd++;
				}
				int indW = k*TileY+j;
				int indR = j*TileX + k;
				unsigned int offsetR = ((i & 1) == 0) ?  DMAX*DMAX/VFACTOR : 0;
				unsigned int offsetW = ((i & 1) == 0) ?  0 : DMAX*DMAX/VFACTOR;
				if(i < B){
					tmp = pipeS::PipeAt<Pidx1>::read();
				}



				tmp_M[indW+offsetW] = tmp;
				struct dPath tmp_R = tmp_M[indR+offsetR];
				if(i > 0){
					pipeS::PipeAt<Pidx2>::write(tmp_R);
				}
			}
		}

		#undef SHIFT
		#undef VFACTOR
		#undef DSIZE
		#undef ADJUST

	});
    });

    // return e;

}

template <size_t idx>  struct undo_interleaved_row_block8_id;
template <bool FPPREC, int DMAX, int Pidx1, int Pidx2>
void undo_interleaved_row_block8(queue &q, ac_int<13,true> M, ac_int<13,true> N, ac_int<15,true> B, ac_int<13,true> n_iter,  bool undo_interleave){


	event e = q.submit([&](handler &h) {
    h.single_task<class undo_interleaved_row_block8_id<Pidx1>>([=] () {
		#define SHIFT (3-FPPREC)
		#define VFACTOR ((1 << SHIFT))
		#define DSIZE 	(256/VFACTOR)
		#define ADJUST (VFACTOR-1)

    	[[intel::disable_loop_pipelining]]
		for(unsigned short u_itr = 0; u_itr < n_iter; u_itr++){

			ac_int<13,true> TileX, TileY;
			ac_int<25,true> NTiles;
			ac_int<9,true> XBlocks = (M >> SHIFT);
			unsigned int offset;
			const int N_CU = VFACTOR;
			switch(undo_interleave){
				case true: {TileX=XBlocks; TileY=N_CU; NTiles = (B*N+ADJUST)>>SHIFT; break;}
				case false: {TileX=8; TileY=N; NTiles = (((XBlocks*B+ADJUST)>>SHIFT)); break;}
				default: {TileX=XBlocks; TileY=N_CU; NTiles = (B*N+ADJUST)>>SHIFT; break;}
			}
			// [[intel::force_pow2_depth(0)]] 
			struct dPath tmp_M[DMAX*N_CU*2];
			#pragma HLS RESOURCE variable=tmp_M core=XPM_MEMORY latency=2
			ac_int<25,true> NTilesp1 = NTiles+1;

			ac_int<25,true> id = 0;
			ac_int<13,true> jd =0, kd = 0;
			int total_itr = NTilesp1*TileX*TileY;

			[[intel::initiation_interval(1)]]
			for(int itr= 0; itr < total_itr; itr++){

				ac_int<25, true> i = id;
				ac_int<13, true> j = jd, k = kd;

				if(k == TileX -1){
					kd = 0;
				} else {
					kd++;
				}

				if(k == TileX -1 && j == TileY -1){
					jd = 0;
					id++;
				} else if(k == TileX -1){
					jd++;
				}
				unsigned int offsetR = ((i & 1) == 0) ?  DMAX*N_CU : 0;
				unsigned int offsetW = ((i & 1) == 0) ?  0 : DMAX*N_CU;

				bool cmpW = !undo_interleave || (i*TileY + j < B*N);
				int indW = j*TileX + k + offsetW;
				struct dPath tmpW;
				if(cmpW && i < NTiles){
					tmpW = pipeS::PipeAt<Pidx1>::read();
				}
				tmp_M[indW] = tmpW;

				int indR = k*TileY+j + offsetR;
				struct dPath tmpR = tmp_M[indR];
				if(i > 0){
					pipeS::PipeAt<Pidx2>::write(tmpR);
				}

				if( Pidx1 == 26){
					PRINTF("undo_interleaved_row_block8 u_iter itr : %d %d\n", u_itr, itr);
				}
			}
		}

		#undef SHIFT
		#undef VFACTOR
		#undef DSIZE
		#undef ADJUST

	});
    });

    // return e;

}


template <size_t idx>  struct col2row_id;
template <bool FPPREC, int DMAX, int Pidx1, int Pidx2>
void col2row(queue &q, ac_int<13,true> M, ac_int<13,true> N, ac_int<15,true> B, ac_int<13,true> n_iter){

	event e = q.submit([&](handler &h) {
    h.single_task<class col2row_id<Pidx1>>([=] () {

		#define SHIFT (3-FPPREC)
		#define VFACTOR ((1 << SHIFT))
		#define DSIZE 	(256/VFACTOR)
		#define ADJUST (VFACTOR-1)

    	[[intel::disable_loop_pipelining]]
		for(unsigned short u_itr = 0; u_itr < n_iter; u_itr++){

			ac_int<13,true> TileX, TileY;
			ac_int<19,true> NTiles;
			ac_int<9,true> XBlocks = (M >> SHIFT);
			unsigned int offset;

			TileX = XBlocks;
			TileY = N;
			NTiles = B;


			// [[intel::force_pow2_depth(0)]] 
			struct dPath tmp_M[DMAX*DMAX/VFACTOR*2];
			struct dPath tmp;

			ac_int<19,true> NTilesp1 = B+1;
			ac_int<19,true> id = 0;
			ac_int<13,true> jd =0, kd = 0;
			int total_itr = NTilesp1*TileX*TileY;

			[[intel::initiation_interval(1)]]
			for(int itr= 0; itr < total_itr; itr++){

				ac_int<19,true> i = id;
				ac_int<13,true> j = jd, k = kd;

				if(k == TileX -1){
					kd = 0;
				} else {
					kd++;
				}

				if(k == TileX -1 && j == TileY -1){
					jd = 0;
					id++;
				} else if(k == TileX -1){
					jd++;
				}

				int indW = j*TileX + k;
				int indR = k*TileY+j;
				unsigned int offsetR = ((i & 1) == 0) ?  DMAX*DMAX/VFACTOR : 0;
				unsigned int offsetW = ((i & 1) == 0) ?  0 : DMAX*DMAX/VFACTOR;

				if(i < B){
					tmp = pipeS::PipeAt<Pidx1>::read();
				}

				tmp_M[indW+offsetW] = tmp;
				struct dPath tmp_R = tmp_M[indR+offsetR];
				if(i > 0){
					pipeS::PipeAt<Pidx2>::write(tmp_R);
				}
			}
		}

		#undef SHIFT
		#undef VFACTOR
		#undef DSIZE
		#undef ADJUST

	});
    });

    // return e;
}


template <size_t idx>  struct stream_8x8transpose_id;
template <bool FPPREC, class DType, int Pidx1, int Pidx2>
void stream_8x8transpose(queue &q, ac_int<13,true> M, ac_int<13,true> N, ac_int<15,true> B,  ac_int<13,true> n_iter,  bool transpose){


	event e = q.submit([&](handler &h) {
    h.single_task<class stream_8x8transpose_id<Pidx1>>([=] () {
		#define SHIFT (3-FPPREC)
		#define VFACTOR ((1 << SHIFT))
		#define DSIZE 	(256/VFACTOR)
		#define ADJUST (VFACTOR-1)

    	[[intel::disable_loop_pipelining]]
    	for(unsigned short u_itr = 0; u_itr < n_iter; u_itr++){

			const int l_interval = VFACTOR;

			ac_int<13,true> TileX, TileY;
			ac_int<25,true> NTiles;
			ac_int<9,true> XBlocks = (M >> SHIFT);
			const int N_CU = FPPREC? 4 :8;

			switch(transpose){
				case true: {TileX=XBlocks; TileY=N_CU; NTiles = ((B*N+ADJUST)>>SHIFT)*XBlocks; break;}
				case false: {TileX=8; TileY=N; NTiles = (((XBlocks*B+ADJUST)>>SHIFT))*N; break;}
				default: {TileX=XBlocks; TileY=N_CU; NTiles = ((B*N+ADJUST)>>SHIFT)*XBlocks; break;}
			}

			
			unsigned int total_itr = (NTiles+1) << SHIFT;
			
			struct dPath tmp[VFACTOR*2];
			struct dPath outR[VFACTOR];
			[[intel::initiation_interval(1)]]
			for(int itr=0; itr < total_itr; itr++){
				
				int blk = (itr >> SHIFT);
				int offset_W = ((blk & 1) == 0 ? 0 : VFACTOR);
				int offset_R = ((blk & 1) == 1 ? 0 : VFACTOR);

				int add_W = (itr & 7) + offset_W;
				int add_R = (itr & 7) + offset_R;
				if (blk < NTiles){
					tmp[add_W] = pipeS::PipeAt<Pidx1>::read();
				}

				#pragma unroll
				for(int v = 0; v < VFACTOR; v++){
					outR[v].data[0] = tmp[offset_R+0].data[v];
					outR[v].data[1] = tmp[offset_R+1].data[v];
					outR[v].data[2] = tmp[offset_R+2].data[v];
					outR[v].data[3] = tmp[offset_R+3].data[v];
					if(!FPPREC){
						outR[v].data[4] = tmp[offset_R+4].data[v];
						outR[v].data[5] = tmp[offset_R+5].data[v];
						outR[v].data[6] = tmp[offset_R+6].data[v];
						outR[v].data[7] = tmp[offset_R+7].data[v];
					}
				}

				struct dPath vec_W = transpose ? outR[itr&7] : tmp[add_R];
				if (blk > 0){
					pipeS::PipeAt<Pidx2>::write(vec_W);
				}


				// if( Pidx1 == 25){
					// PRINTF("stream_8x8transpose u_iter itr : %d %d\n", u_itr, itr);
				// }
				

			}
		}



		#undef SHIFT
		#undef VFACTOR
		#undef DSIZE
		#undef ADJUST


	});
    });

    // return e;

}


// template <int MEM_SIZE, int>
// static void URAM_buffer(queue &q, int total_data, ap_uint<20> delay){

// 	struct dPath mem[MEM_SIZE];
// 	int total_itr = total_data + delay;
// 	ap_uint<20> count = 0;

// 	for(int i = 0; i < total_itr; i++){
// 		#pragma HLS loop_tripcount min=500000 max=2000000 avg=2000000
// 		#pragma HLS PIPELINE II=1

// 		if(i >= delay){
// 			out_stm_1 << mem[count];
// 		}

// 		if(i < total_data){
// 			mem[count] = in_stm_1.read();
// 		}

// 		if(count >= delay -1){
// 			count = 0;
// 		} else{
// 			count++;
// 		}
// 	}
// }



#endif


