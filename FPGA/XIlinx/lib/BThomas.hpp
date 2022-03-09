
#include <ap_int.h>
#include <hls_stream.h>
#include <data_types.h>

#ifndef __BTHOMAS_H__
#define __BTHOMAS_H__

template<bool FPPREC, class DType, int DMAX>
static void thomas_interleave(hls::stream<uint256_dt> &d_stm, hls::stream<uint256_dt> &d_fw_stm,
		ap_uint<12> d0, ap_uint<14> B, int ReadLimit){


	#define SHIFT (3-FPPREC)
	#define VFACTOR ((1 << SHIFT))
	#define DSIZE 	(256/VFACTOR)
	#define NBLK 	(32<<FPPREC)
	const int n_blk = NBLK;

	uint256_dt  d2[DMAX*NBLK*2];
	#pragma HLS RESOURCE variable=d2 core=XPM_MEMORY uram latency=2
	ap_uint<16> batd1 = 0;
	ap_uint<7> id1 =0;
	ap_uint<12> jd1 = 0;
	ap_uint<16> Bp1 = B+1;

	int total_itr =register_it<int> (B*NBLK*d0 + NBLK*d0);
	loop_read: for(int itr= 0; itr < total_itr; itr++){
			#pragma HLS PIPELINE II=1
			#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000

			ap_uint<16> bat = batd1;
			ap_uint<7> i = id1;
			ap_uint<12> j = jd1;

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

			ap_uint<20> countr1 = register_it<int>((bat*NBLK) + i);
			int count = countr1 * d0 + j;
			uint256_dt  tmp_d;
			if(count < ReadLimit){
				tmp_d = d_stm.read();
			} else {
				tmp_d = 0;
			}
			int indW = j*NBLK + i + offsetW;
			d2[indW] = tmp_d;

			int indR = i*d0+j + offsetR;
			uint256_dt  tmp_R = d2[indR];
			if(bat > 0){
				d_fw_stm << tmp_R;
			}
		}

	#undef SHIFT
	#undef VFACTOR
	#undef DSIZE
	#undef NBLK
}

template<bool FPPREC, class DType, int DMAX>
static void thomas_forward(hls::stream<uint256_dt> &d_fw_stm, hls::stream<uint256_dt> &c2_fw_stm, hls::stream<uint256_dt> &d2_fw_stm,
		ap_uint<12> d0, ap_uint<14> B){


	#define SHIFT (3-FPPREC)
	#define VFACTOR ((1 << SHIFT))
	#define DSIZE 	(256/VFACTOR)
	#define NBLK 	(32<<FPPREC)
	const int n_blk = NBLK;

	uint256_dt  c2_fw[DMAX*NBLK*2];
	uint256_dt  d2_fw[DMAX*NBLK*2];

	#pragma HLS RESOURCE variable=c2_fw core=XPM_MEMORY uram latency=2
	#pragma HLS RESOURCE variable=d2_fw core=XPM_MEMORY uram latency=2


	ap_uint<16> batd2 = 0;
	ap_uint<12> id2 =0;
	ap_uint<7>  kd2 = 0;

	ap_uint<16> Bp1 = B+1;
	int total_itr =register_it<int> (B*NBLK*d0 + NBLK*d0);

	uint256_dt window_b2[NBLK], window_c2[NBLK], window_d2[NBLK];
	loop_fw: for(int itr= 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000

		ap_uint<16> bat = batd2;
		ap_uint<12> i = id2;
		ap_uint<7> k = kd2;

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

		#pragma HLS dependence variable=window_b2 RAW distance=n_blk true
		#pragma HLS dependence variable=window_d2 RAW distance=n_blk true
		#pragma HLS dependence variable=window_c2 RAW distance=n_blk true

		uint256_dt d2_read = 0;
		if(bat < B){
			d2_read = d_fw_stm.read();
		}

		uint256_dt vec_bb_r = window_b2[k];
		uint256_dt vec_dd_r = window_d2[k];
		uint256_dt vec_cc_r = window_c2[k];

		uint256_dt b2_fw_write, d2_fw_write;
		uint256_dt vec_bb_w, vec_dd_w, vec_cc_w;

		fw_vec_loop: for(int v =0; v < VFACTOR; v++){
			#pragma HLS unroll
			DType aa_read = (i == 0 || i == d0 -1) ? 0.0 : -0.5; // uint2FP_ript(a2_read.range(D_SIZE*(v+1)-1,D_SIZE*v));
			DType bb_read = (i == 0 || i == d0 -1) ? 1.0 :  2.0; //uint2FP_ript(b2_read.range(D_SIZE*(v+1)-1,D_SIZE*v));
			DType cc_read = (i == 0 || i == d0 -1) ? 0.0 : -0.5; //uint2FP_ript(c2_read.range(D_SIZE*(v+1)-1,D_SIZE*v));
			DType dd_read = uint2FP_ript<FPPREC, DType>(d2_read.range(DSIZE*(v+1)-1,DSIZE*v));

			DType bb_old = uint2FP_ript<FPPREC, DType>(vec_bb_r.range(DSIZE*(v+1)-1,DSIZE*v));
			DType dd_old = uint2FP_ript<FPPREC, DType>(vec_dd_r.range(DSIZE*(v+1)-1,DSIZE*v));
			DType cc_old = uint2FP_ript<FPPREC, DType>(vec_cc_r.range(DSIZE*(v+1)-1,DSIZE*v));



			DType denom = (i == 0) ? bb_read : (bb_read - aa_read*cc_old);
			DType r = 1.0/denom;
			DType c_w1 = cc_read;
			DType d_w1 = (i == 0) ? dd_read : (dd_read - aa_read*dd_old);

			DType b_wr = 1.0;
			DType c_wr = c_w1*r;
			DType d_wr = d_w1*r;



			b2_fw_write.range(DSIZE*(v+1)-1,DSIZE*v) = FP2uint_ript(b_wr);
			d2_fw_write.range(DSIZE*(v+1)-1,DSIZE*v) = FP2uint_ript(d_wr);

			vec_bb_w.range(DSIZE*(v+1)-1,DSIZE*v) = FP2uint_ript(b_wr);
			vec_dd_w.range(DSIZE*(v+1)-1,DSIZE*v) = FP2uint_ript(d_wr);
			vec_cc_w.range(DSIZE*(v+1)-1,DSIZE*v) = FP2uint_ript(c_wr);

		}
		window_b2[k] = vec_bb_w;
		window_d2[k] = vec_dd_w;
		window_c2[k] = vec_cc_w;
		unsigned int offsetR = ((bat & 1) == 0) ?  DMAX*NBLK : 0;
		unsigned int offsetW = ((bat & 1) == 0) ?  0 : DMAX*NBLK;
		int indW =  k*d0+i+offsetW;
		c2_fw[indW] = vec_cc_w;
		d2_fw[indW] = d2_fw_write;
		ap_uint<12> i_rev = d0-i -1;
		int indR =  k*d0+ i_rev + offsetR;

		uint256_dt c2_fw_stmR = c2_fw[indR];
		uint256_dt d2_fw_stmR = d2_fw[indR];

		if(bat > 0){
			c2_fw_stm << c2_fw_stmR;
			d2_fw_stm << d2_fw_stmR;
		}

	}

	#undef SHIFT
	#undef VFACTOR
	#undef DSIZE
	#undef NBLK
}



template<bool FPPREC, class DType, int DMAX>
static void thomas_forward(hls::stream<uint256_dt> &d_fw_stm, hls::stream<uint256_dt> &c2_fw_stm, hls::stream<uint256_dt> &d2_fw_stm,
		DType coeff_a(int itr, int sys_element, int sys_index, int bat, int sys_size), DType coeff_b(int itr, int sys_element, int sys_index, int bat, int sys_size), 
		DType coeff_c(int itr, int sys_element, int sys_index, int bat, int sys_size), 
		ap_uint<12> d0, ap_uint<14> B){


	#define SHIFT (3-FPPREC)
	#define VFACTOR ((1 << SHIFT))
	#define DSIZE 	(256/VFACTOR)
	#define NBLK 	(32<<FPPREC)
	const int n_blk = NBLK;

	uint256_dt  c2_fw[DMAX*NBLK*2];
	uint256_dt  d2_fw[DMAX*NBLK*2];

	#pragma HLS RESOURCE variable=c2_fw core=XPM_MEMORY uram latency=2
	#pragma HLS RESOURCE variable=d2_fw core=XPM_MEMORY uram latency=2


	ap_uint<16> batd2 = 0;
	ap_uint<12> id2 =0;
	ap_uint<7>  kd2 = 0;

	ap_uint<16> Bp1 = B+1;
	int total_itr =register_it<int> (B*NBLK*d0 + NBLK*d0);

	uint256_dt window_b2[NBLK], window_c2[NBLK], window_d2[NBLK];
	loop_fw: for(int itr= 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000

		ap_uint<16> bat = batd2;
		ap_uint<12> i = id2;
		ap_uint<7> k = kd2;

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

		#pragma HLS dependence variable=window_b2 RAW distance=n_blk true
		#pragma HLS dependence variable=window_d2 RAW distance=n_blk true
		#pragma HLS dependence variable=window_c2 RAW distance=n_blk true

		uint256_dt d2_read = 0;
		if(bat < B){
			d2_read = d_fw_stm.read();
		}

		uint256_dt vec_bb_r = window_b2[k];
		uint256_dt vec_dd_r = window_d2[k];
		uint256_dt vec_cc_r = window_c2[k];

		uint256_dt b2_fw_write, d2_fw_write;
		uint256_dt vec_bb_w, vec_dd_w, vec_cc_w;

		fw_vec_loop: for(int v =0; v < VFACTOR; v++){
			#pragma HLS unroll
			DType aa_read = coeff_a(itr, i, k, bat); // (i == 0 || i == d0 -1) ? 0.0 : -0.5; // uint2FP_ript(a2_read.range(D_SIZE*(v+1)-1,D_SIZE*v));
			DType bb_read = coeff_b(itr, i, k, bat); // (i == 0 || i == d0 -1) ? 1.0 :  2.0; //uint2FP_ript(b2_read.range(D_SIZE*(v+1)-1,D_SIZE*v));
			DType cc_read = coeff_c(itr, i, k, bat); // (i == 0 || i == d0 -1) ? 0.0 : -0.5; //uint2FP_ript(c2_read.range(D_SIZE*(v+1)-1,D_SIZE*v));
			DType dd_read = uint2FP_ript<FPPREC, DType>(d2_read.range(DSIZE*(v+1)-1,DSIZE*v));

			DType bb_old = uint2FP_ript<FPPREC, DType>(vec_bb_r.range(DSIZE*(v+1)-1,DSIZE*v));
			DType dd_old = uint2FP_ript<FPPREC, DType>(vec_dd_r.range(DSIZE*(v+1)-1,DSIZE*v));
			DType cc_old = uint2FP_ript<FPPREC, DType>(vec_cc_r.range(DSIZE*(v+1)-1,DSIZE*v));



			DType denom = (i == 0) ? bb_read : (bb_read - aa_read*cc_old);
			DType r = 1.0/denom;
			DType c_w1 = cc_read;
			DType d_w1 = (i == 0) ? dd_read : (dd_read - aa_read*dd_old);

			DType b_wr = 1.0;
			DType c_wr = c_w1*r;
			DType d_wr = d_w1*r;



			b2_fw_write.range(DSIZE*(v+1)-1,DSIZE*v) = FP2uint_ript(b_wr);
			d2_fw_write.range(DSIZE*(v+1)-1,DSIZE*v) = FP2uint_ript(d_wr);

			vec_bb_w.range(DSIZE*(v+1)-1,DSIZE*v) = FP2uint_ript(b_wr);
			vec_dd_w.range(DSIZE*(v+1)-1,DSIZE*v) = FP2uint_ript(d_wr);
			vec_cc_w.range(DSIZE*(v+1)-1,DSIZE*v) = FP2uint_ript(c_wr);

		}
		window_b2[k] = vec_bb_w;
		window_d2[k] = vec_dd_w;
		window_c2[k] = vec_cc_w;
		unsigned int offsetR = ((bat & 1) == 0) ?  DMAX*NBLK : 0;
		unsigned int offsetW = ((bat & 1) == 0) ?  0 : DMAX*NBLK;
		int indW =  k*d0+i+offsetW;
		c2_fw[indW] = vec_cc_w;
		d2_fw[indW] = d2_fw_write;
		ap_uint<12> i_rev = d0-i -1;
		int indR =  k*d0+ i_rev + offsetR;

		uint256_dt c2_fw_stmR = c2_fw[indR];
		uint256_dt d2_fw_stmR = d2_fw[indR];

		if(bat > 0){
			c2_fw_stm << c2_fw_stmR;
			d2_fw_stm << d2_fw_stmR;
		}

	}

	#undef SHIFT
	#undef VFACTOR
	#undef DSIZE
	#undef NBLK
}


template<bool FPPREC, class DType, int DMAX>
static void thomas_backward(hls::stream<uint256_dt> &c2_fw_stm, hls::stream<uint256_dt> &d2_fw_stm, hls::stream<uint256_dt> &u_stm,
		ap_uint<12> d0, ap_uint<14> B, int ReadLimit){

	#define SHIFT (3-FPPREC)
	#define VFACTOR ((1 << SHIFT))
	#define DSIZE 	(256/VFACTOR)
	#define NBLK 	(32<<FPPREC)
	const int n_blk = NBLK;

	uint256_dt  u2[DMAX*NBLK*2];
	#pragma HLS RESOURCE variable=u2 core=XPM_MEMORY uram = latency=2

	uint256_dt window_u2[NBLK];

	ap_uint<16> batd3 = 0;
	ap_uint<12> id3 =0;
	ap_uint<7>  kd3 = 0;

	ap_uint<16> Bp1 = B+1;
	int total_itr =register_it<int> (B*NBLK*d0 + NBLK*d0);

	loop_bw: for(int itr= 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000

		ap_uint<16> bat = batd3;
		ap_uint<12> id = id3;
		ap_uint<7> k = kd3;

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

		#pragma HLS dependence variable=window_u2 RAW distance=n_blk true
		ap_uint<12> i = d0 -1 -id;

		uint256_dt d2_fw_read = 0;
		uint256_dt c2_fw_read = 0;
		if(bat < B){
			d2_fw_read = d2_fw_stm.read();
			c2_fw_read = c2_fw_stm.read();
		}
		uint256_dt u2_write;

		uint256_dt vec_u2_r = window_u2[k];
		uint256_dt vec_u2_w;
		bw_vec_loop: for(int v = 0; v < VFACTOR; v++){
			#pragma HLS unroll
			DType dd_read = uint2FP_ript<FPPREC>(d2_fw_read.range(DSIZE*(v+1)-1,DSIZE*v));
			DType cc_read = uint2FP_ript<FPPREC>(c2_fw_read.range(DSIZE*(v+1)-1,DSIZE*v));

			DType u_pre = uint2FP_ript<FPPREC>(vec_u2_r.range(DSIZE*(v+1)-1,DSIZE*v));
			DType numer_l = dd_read;
			DType numer_o = (dd_read - cc_read * u_pre);
			DType numer = (i == d0-1) ? numer_l : numer_o;

			DType u_new = numer;
			u2_write.range(DSIZE*(v+1)-1,DSIZE*v) = FP2uint_ript(u_new);
			vec_u2_w.range(DSIZE*(v+1)-1,DSIZE*v) = FP2uint_ript(u_new);
		}

		unsigned int offsetR = ((bat & 1) == 0) ?  DMAX*NBLK : 0;
		unsigned int offsetW = ((bat & 1) == 0) ?  0 : DMAX*NBLK;

		int indW = k* d0 + i + offsetW;
		int indR = id*NBLK+k + offsetR;

		u2[indW] = u2_write;
		uint256_dt u_stm_R = u2[indR];
		int count = (bat-1) * NBLK * d0 + id*NBLK+k;
		if(bat >0 && count < ReadLimit){
			u_stm << u_stm_R;
		}
		window_u2[k] = vec_u2_w;
	}

	#undef SHIFT
	#undef VFACTOR
	#undef DSIZE
	#undef NBLK

}

#endif
