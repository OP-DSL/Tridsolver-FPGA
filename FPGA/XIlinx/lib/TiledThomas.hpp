#include <ap_int.h>
#include <hls_stream.h>
#include <data_types.h>

#ifndef __DTILED_THOMAS_H__
#define __DTILED_THOMAS_H__


// function declarations
template <bool FPPREC, class DType, int NMAX>
static void TT_Interleave( hls::stream<uint256_dt> &d_stm_in,
		hls::stream<uint256_dt> &a_stm_out, hls::stream<uint256_dt> &b_stm_out, hls::stream<uint256_dt> &c_stm_out, hls::stream<uint256_dt> &d_stm_out,
		ap_uint<12> B_size, ap_uint<6> Tiles, ap_uint<12> M, unsigned short Sys, ap_uint<8> Ti, unsigned int ReadLimit);


template <bool FPPREC, class DType, int NMAX>
static void TT_ForwardSweep(hls::stream<uint256_dt> &a_stm_in, hls::stream<uint256_dt> &b_stm_in,  hls::stream<uint256_dt> &c_stm_in, hls::stream<uint256_dt> &d_stm_in,
		hls::stream<uint256_dt> &a_stm_out, hls::stream<uint256_dt> &c_stm_out, hls::stream<uint256_dt> &d_stm_out,
		ap_uint<12> B_size, ap_uint<6> Tiles, ap_uint<12> M, unsigned short Sys, ap_uint<8> Ti);


template <bool FPPREC, class DType, int NMAX>
static void TT_BackwardSweep(hls::stream<uint256_dt> &a_stm_in, hls::stream<uint256_dt> &c_stm_in, hls::stream<uint256_dt> &d_stm_in,
		hls::stream<uint256_dt> &a_stm_out, hls::stream<uint256_dt> &c_stm_out, hls::stream<uint256_dt> &d_stm_out,
		hls::stream<uint256_dt> &ra_stm_out, hls::stream<uint256_dt> &rb_stm_out, hls::stream<uint256_dt> &rc_stm_out, hls::stream<uint256_dt> &rd_stm_out,
		ap_uint<12> B_size, ap_uint<6> Tiles, ap_uint<12> M, unsigned short Sys, ap_uint<8> Ti);


template <bool FPPREC, class DType, int NMAX>
static void TT_BackSubstitution(hls::stream<uint256_dt> &a_stm_in, hls::stream<uint256_dt> &c_stm_in, hls::stream<uint256_dt> &d_stm_in,
		hls::stream<uint256_dt> &u_top_in, hls::stream<uint256_dt> &u_bottom_in, hls::stream<uint256_dt> &u_out,
		ap_uint<22> B_size, ap_uint<6> Tiles, ap_uint<12> M, unsigned int ReadLimit);




// Tiled Solver support definitions

template <bool FPPREC, class DType, int NMAX>
static void TT_Interleave( hls::stream<uint256_dt> &d_stm_in,
		hls::stream<uint256_dt> &a_stm_out, hls::stream<uint256_dt> &b_stm_out, hls::stream<uint256_dt> &c_stm_out, hls::stream<uint256_dt> &d_stm_out,
		ap_uint<12> B_size, ap_uint<6> Tiles, ap_uint<12> M, unsigned short Sys, ap_uint<8> Ti, unsigned int ReadLimit
		){


	#define SHIFT (3-FPPREC)
	#define VFACTOR ((1 << SHIFT))
	#define DSIZE 	(256/VFACTOR)


	uint256_dt  a_pre[NMAX], b_pre[NMAX], c_pre[NMAX], d_pre[NMAX];
	#pragma HLS RESOURCE variable=a_pre core=XPM_MEMORY uram
	#pragma HLS RESOURCE variable=b_pre core=XPM_MEMORY uram
	#pragma HLS RESOURCE variable=c_pre core=XPM_MEMORY uram
	#pragma HLS RESOURCE variable=d_pre core=XPM_MEMORY uram

	ap_uint<22> batL = B_size+1;
	ap_uint<22> batd = 0;
	ap_uint<6> id =0;
	ap_uint<12> jd = 0;
	int total_itr = batL*Tiles*M;
	for(int itr= 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000

		ap_uint<22> bat = batd;
		ap_uint<6> i = id;
		ap_uint<12> j = jd;

		if(j == M -1){
			jd = 0;
		} else {
			jd++;
		}

		if(j == M -1 && i == Tiles -1){
			id = 0;
			batd++;
		} else if(j == M -1){
			id++;
		}


		uint256_dt tmp_a, tmp_b, tmp_c, tmp_d;
		int count = bat * Tiles*M + i * M + j;
		if(count < ReadLimit && bat < B_size){
			tmp_d = d_stm_in.read();
		} else {
			tmp_a = 0;
			tmp_b = 0;
			tmp_c = 0;
			tmp_d = 0;
		}

		DType tmp_a_f[8], tmp_b_f[8], tmp_c_f[8], tmp_d_f[8];
		uint256_dt tmp_a_w, tmp_b_w, tmp_c_w, tmp_d_w;
		for(int v = 0; v < VFACTOR; v++){
			tmp_a_f[v] = uint2FP_ript<FPPREC, DType>(tmp_a.range(DSIZE*(v+1)-1,DSIZE*v));
			tmp_b_f[v] = uint2FP_ript<FPPREC, DType>(tmp_b.range(DSIZE*(v+1)-1,DSIZE*v));
			tmp_c_f[v] = uint2FP_ript<FPPREC, DType>(tmp_c.range(DSIZE*(v+1)-1,DSIZE*v));
			tmp_d_f[v] = uint2FP_ript<FPPREC, DType>(tmp_d.range(DSIZE*(v+1)-1,DSIZE*v));
			ap_uint<12> d0 = (i % Ti)*M + j;
			ap_uint<12> sys_size = Ti*M;


			tmp_a_f[v] = (d0 == 0 || d0 == sys_size-1) ? 0.0f : -0.5f;
			tmp_c_f[v] = (d0 == 0 || d0 == sys_size-1) ? 0.0f : -0.5f;
			tmp_b_f[v] = (d0 == 0 || d0 == sys_size-1) ? 1.0f :  2.0f;

			tmp_a_w.range(DSIZE*(v+1)-1,DSIZE*v) = FP2uint_ript(tmp_a_f[v]);
			tmp_b_w.range(DSIZE*(v+1)-1,DSIZE*v) = FP2uint_ript(tmp_b_f[v]);
			tmp_c_w.range(DSIZE*(v+1)-1,DSIZE*v) = FP2uint_ript(tmp_c_f[v]);
			tmp_d_w.range(DSIZE*(v+1)-1,DSIZE*v) = FP2uint_ript(tmp_d_f[v]);

		}

		int offsetW = ((bat & 1) == 0 ? 0 : NMAX/2);
		int offsetR = ((bat & 1) == 1 ? 0 : NMAX/2);

		int ind_w = j*Tiles+i;
		int ind_R = i*M+j;
		a_pre[ind_w+offsetW] = tmp_a_w;
		b_pre[ind_w+offsetW] = tmp_b_w;
		c_pre[ind_w+offsetW] = tmp_c_w;
		d_pre[ind_w+offsetW] = tmp_d_w;

		uint256_dt tmp_a_R = a_pre[ind_R+offsetR];
		uint256_dt tmp_b_R = b_pre[ind_R+offsetR];
		uint256_dt tmp_c_R = c_pre[ind_R+offsetR];
		uint256_dt tmp_d_R = d_pre[ind_R+offsetR];

		if(bat > 0){
			a_stm_out << tmp_a_R;
			b_stm_out << tmp_b_R;
			c_stm_out << tmp_c_R;
			d_stm_out << tmp_d_R;
		}

	}

	#undef SHIFT
	#undef VFACTOR
	#undef DSIZE
}

template <bool FPPREC, class DType, int NMAX>
static void TT_ForwardSweep(hls::stream<uint256_dt> &a_stm_in, hls::stream<uint256_dt> &b_stm_in,  hls::stream<uint256_dt> &c_stm_in, hls::stream<uint256_dt> &d_stm_in,
		hls::stream<uint256_dt> &a_stm_out, hls::stream<uint256_dt> &c_stm_out, hls::stream<uint256_dt> &d_stm_out,
		ap_uint<12> B_size, ap_uint<6> Tiles, ap_uint<12> M, unsigned short Sys, ap_uint<8> Ti
		){


	#define SHIFT (3-FPPREC)
	#define VFACTOR ((1 << SHIFT))
	#define DSIZE 	(256/VFACTOR)
	#define NBLK 	(32<<FPPREC)
	const int n_blk = NBLK;

	uint256_dt  a_fw[NMAX], c_fw[NMAX], d_fw[NMAX];
	#pragma HLS RESOURCE variable=a_fw core=XPM_MEMORY uram
	#pragma HLS RESOURCE variable=c_fw core=XPM_MEMORY uram
	#pragma HLS RESOURCE variable=d_fw core=XPM_MEMORY uram


	uint256_dt window_a2[NBLK], window_c2[NBLK], window_d2[NBLK];

	ap_uint<22> batL = B_size+1;
	ap_uint<22> batd = 0;
	ap_uint<12> id =0;
	ap_uint<8> kd = 0;
	int total_itr = batL*Tiles*M;
	for(int itr= 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000
		#pragma HLS dependence variable=window_a2 RAW distance=n_blk true
		#pragma HLS dependence variable=window_c2 RAW distance=n_blk true
		#pragma HLS dependence variable=window_d2 RAW distance=n_blk true

		ap_uint<22> bat = batd;
		ap_uint<12> i = id;
		ap_uint<8> k = kd;

		if(k == Tiles -1){
			kd = 0;
		} else {
			kd++;
		}

		if(k == Tiles -1 && i == M -1){
			id = 0;
			batd++;
		} else if(k == Tiles -1){
			id++;
		}

		int ind =  k*M+i;
		uint256_dt a_vec_r, b_vec_r, c_vec_r, d_vec_r;
		if(bat < B_size){
			a_vec_r = a_stm_in.read();
			b_vec_r = b_stm_in.read();
			c_vec_r = c_stm_in.read();
			d_vec_r = d_stm_in.read();
		}

		uint256_dt a_vec_or = window_a2[k];
		uint256_dt c_vec_or = window_c2[k];
		uint256_dt d_vec_or = window_d2[k];

		uint256_dt vec_a_w, vec_d_w, vec_c_w;

		fw_vec_loop_TM: for(int v =0; v < VFACTOR; v++){
			#pragma HLS unroll
			DType a = uint2FP_ript<FPPREC, DType>(a_vec_r.range(DSIZE*(v+1)-1,DSIZE*v));
			DType b = uint2FP_ript<FPPREC, DType>(b_vec_r.range(DSIZE*(v+1)-1,DSIZE*v)); //normalised to one
			DType c = uint2FP_ript<FPPREC, DType>(c_vec_r.range(DSIZE*(v+1)-1,DSIZE*v));
			DType d = uint2FP_ript<FPPREC, DType>(d_vec_r.range(DSIZE*(v+1)-1,DSIZE*v));

			DType a_old = uint2FP_ript<FPPREC, DType>(a_vec_or.range(DSIZE*(v+1)-1,DSIZE*v));
			DType c_old = uint2FP_ript<FPPREC, DType>(c_vec_or.range(DSIZE*(v+1)-1,DSIZE*v));
			DType d_old = uint2FP_ript<FPPREC, DType>(d_vec_or.range(DSIZE*(v+1)-1,DSIZE*v));

			DType denom = (i == 0 || i == 1) ? b : (b - a*c_old);
			DType r = 1/ denom;
			DType d_w1 = (i == 0 || i == 1) ? d : (d - a*d_old);
			DType a_w1 = (i == 0 || i == 1) ? a : (-1.0f)*a*a_old;
			DType c_w1 = (i == 0 || i == 1) ? c : c;

			DType d_w = r*d_w1;
			DType a_w = r*a_w1;
			DType c_w = r*c_w1;

			vec_a_w.range(DSIZE*(v+1)-1,DSIZE*v) = FP2uint_ript(a_w);
			vec_c_w.range(DSIZE*(v+1)-1,DSIZE*v) = FP2uint_ript(c_w);
			vec_d_w.range(DSIZE*(v+1)-1,DSIZE*v) = FP2uint_ript(d_w);

		}

		window_a2[k] = vec_a_w;
		window_c2[k] = vec_c_w;
		window_d2[k] = vec_d_w;

		int offsetW = ((bat & 1) == 0 ? 0 : NMAX/2);
		int offsetR = ((bat & 1) == 1 ? 0 : NMAX/2);

		int ind_R =  k*M+i;
		int ind_W =  k*M+(M-1-i);

		a_fw[ind_R+offsetW] = vec_a_w;
		c_fw[ind_R+offsetW] = vec_c_w;
		d_fw[ind_R+offsetW] = vec_d_w;

		uint256_dt tmp_a_W = a_fw[ind_W+offsetR];
		uint256_dt tmp_c_W = c_fw[ind_W+offsetR];
		uint256_dt tmp_d_W = d_fw[ind_W+offsetR];

		if(bat > 0){
			a_stm_out << tmp_a_W;
			c_stm_out << tmp_c_W;
			d_stm_out << tmp_d_W;
		}

	}

	#undef SHIFT
	#undef VFACTOR
	#undef DSIZE
	#undef NBLK

}


template <bool FPPREC, class DType, int NMAX>
static void TT_BackwardSweep(hls::stream<uint256_dt> &a_stm_in, hls::stream<uint256_dt> &c_stm_in, hls::stream<uint256_dt> &d_stm_in,
		hls::stream<uint256_dt> &a_stm_out, hls::stream<uint256_dt> &c_stm_out, hls::stream<uint256_dt> &d_stm_out,
		hls::stream<uint256_dt> &ra_stm_out, hls::stream<uint256_dt> &rb_stm_out, hls::stream<uint256_dt> &rc_stm_out, hls::stream<uint256_dt> &rd_stm_out,
		ap_uint<12> B_size, ap_uint<6> Tiles, ap_uint<12> M, unsigned short Sys, ap_uint<8> Ti
		){


	#define SHIFT (3-FPPREC)
	#define VFACTOR ((1 << SHIFT))
	#define DSIZE 	(256/VFACTOR)
	#define NBLK 	(32<<FPPREC)
	#define RNMAX 	512
	const int n_blk = NBLK;

	uint256_dt  a_bw[NMAX], c_bw[NMAX], d_bw[NMAX];
	uint256_dt  ra_pre[RNMAX], rb_pre[RNMAX], rc_pre[RNMAX], rd_pre[RNMAX];
	#pragma HLS RESOURCE variable=a_bw core=XPM_MEMORY uram
	#pragma HLS RESOURCE variable=c_bw core=XPM_MEMORY uram
	#pragma HLS RESOURCE variable=d_bw core=XPM_MEMORY uram

	uint256_dt window_a_RTM[NBLK], window_c_RTM[NBLK], window_d_RTM[NBLK];

	ap_uint<22> batL = B_size+1;
	ap_uint<22> batd = 0;
	ap_uint<12> idd =0;
	ap_uint<8> kd = 0;
	int total_itr = batL*Tiles*M;
	for(int itr= 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000
		#pragma HLS dependence variable=window_a_RTM RAW distance=n_blk true
		#pragma HLS dependence variable=window_c_RTM RAW distance=n_blk true
		#pragma HLS dependence variable=window_d_RTM RAW distance=n_blk true

		ap_uint<22> bat = batd;
		ap_uint<12> id = idd;
		ap_uint<8> k = kd;

		if(k == Tiles -1){
			kd = 0;
		} else {
			kd++;
		}

		if(k == Tiles -1 && id == M -1){
			idd = 0;
			batd++;
		} else if(k == Tiles -1){
			idd++;
		}

		ap_uint<12> i = M - 1 -id;
		int ind =  k*M+i;
		uint256_dt a_vec_r, c_vec_r, d_vec_r;
		if(bat < B_size){
			a_vec_r = a_stm_in.read();
			c_vec_r = c_stm_in.read();
			d_vec_r = d_stm_in.read();
		}

		uint256_dt a_vec_or = window_a_RTM[k];
		uint256_dt c_vec_or = window_c_RTM[k];
		uint256_dt d_vec_or = window_d_RTM[k];

		uint256_dt a_vec_w, b_vec_w, c_vec_w, d_vec_w;
		bw_vec_loop_TM: for(int v = 0; v < VFACTOR; v++){

			DType a_r = uint2FP_ript<FPPREC, DType>(a_vec_r.range(DSIZE*(v+1)-1,DSIZE*v));
			DType c_r = uint2FP_ript<FPPREC, DType>(c_vec_r.range(DSIZE*(v+1)-1,DSIZE*v));
			DType d_r = uint2FP_ript<FPPREC, DType>(d_vec_r.range(DSIZE*(v+1)-1,DSIZE*v));

			DType a_or = uint2FP_ript<FPPREC, DType>(a_vec_or.range(DSIZE*(v+1)-1,DSIZE*v));
			DType c_or = uint2FP_ript<FPPREC, DType>(c_vec_or.range(DSIZE*(v+1)-1,DSIZE*v));
			DType d_or = uint2FP_ript<FPPREC, DType>(d_vec_or.range(DSIZE*(v+1)-1,DSIZE*v));

			DType d_w = (i == M-1 || i == M-2) ? d_r : d_r - c_r * d_or;
			DType a_w = (i == M-1 || i == M-2 || i == 0) ?a_r : a_r - c_r*a_or;
			DType b_w = (i == 0) ? 1.0f - c_r*a_or : 1.0f;
			DType c_w = (i == M-1 || i == M-2) ? c_r : -c_r * c_or;

			a_vec_w.range(DSIZE*(v+1)-1,DSIZE*v) = FP2uint_ript(a_w);
			b_vec_w.range(DSIZE*(v+1)-1,DSIZE*v) = FP2uint_ript(b_w);
			c_vec_w.range(DSIZE*(v+1)-1,DSIZE*v) = FP2uint_ript(c_w);
			d_vec_w.range(DSIZE*(v+1)-1,DSIZE*v) = FP2uint_ript(d_w);

		}

		int offsetW = ((bat & 1) == 0 ? 0 : NMAX/2);
		int offsetR = ((bat & 1) == 1 ? 0 : NMAX/2);

		int offsetW_rr = ((bat & 1) == 0 ? 0 : RNMAX/2);
		int offsetR_rr = ((bat & 1) == 1 ? 0 : RNMAX/2);

		unsigned char ind_r;
		if(i == 0){
			ind_r = (k << 1);
		} else {
			ind_r = (k << 1) + 1;
		}
		unsigned char k_i = ind_r % (Ti<<1);
		unsigned char sys_i = ind_r / (Ti<<1);

		if(i == 0 || i == M-1){
			int ind_rW = k_i*Sys + sys_i;

			ra_pre[ind_rW+offsetW_rr] = a_vec_w;
			rb_pre[ind_rW+offsetW_rr] = b_vec_w;
			rc_pre[ind_rW+offsetW_rr] = c_vec_w;
			rd_pre[ind_rW+offsetW_rr] = d_vec_w;
		}

		unsigned char ind_rr = (M-1-i)*Tiles + k;
		uint256_dt tmp_ra_preR = ra_pre[ind_rr+offsetR_rr];
		uint256_dt tmp_rb_preR = rb_pre[ind_rr+offsetR_rr];
		uint256_dt tmp_rc_preR = rc_pre[ind_rr+offsetR_rr];
		uint256_dt tmp_rd_preR = rd_pre[ind_rr+offsetR_rr];

		if((i == M-1 || i == M-2) && bat > 0){
			ra_stm_out << tmp_ra_preR;
			rb_stm_out << tmp_rb_preR;
			rc_stm_out << tmp_rc_preR;
			rd_stm_out << tmp_rd_preR;
		}

		window_a_RTM[k] = a_vec_w;
		window_c_RTM[k] = c_vec_w;
		window_d_RTM[k] = d_vec_w;

		int ind_W = k*M+i;
		a_bw[ind_W+offsetW] = a_vec_w;
		c_bw[ind_W+offsetW] = c_vec_w;
		d_bw[ind_W+offsetW] = d_vec_w;


		int ind_rev = (M-1-i)*Tiles + k;
		uint256_dt tmp_a_preR = a_bw[ind_rev+offsetR];
		uint256_dt tmp_c_preR = c_bw[ind_rev+offsetR];
		uint256_dt tmp_d_preR = d_bw[ind_rev+offsetR];

		if(bat > 0){
			a_stm_out << tmp_a_preR;
			c_stm_out << tmp_c_preR;
			d_stm_out << tmp_d_preR;
		}

	}

	#undef SHIFT
	#undef VFACTOR
	#undef DSIZE
	#undef NBLK
	#undef RNMAX

//	printf("Stage 2 has been returned\n");
}


template <bool FPPREC, class DType, int NMAX>
static void TT_BackSubstitution(hls::stream<uint256_dt> &a_stm_in, hls::stream<uint256_dt> &c_stm_in, hls::stream<uint256_dt> &d_stm_in,
		hls::stream<uint256_dt> &u_top_in, hls::stream<uint256_dt> &u_bottom_in, hls::stream<uint256_dt> &u_out,
		ap_uint<22> B_size, ap_uint<6> Tiles, ap_uint<12> M, unsigned int ReadLimit
		){

	#define SHIFT (3-FPPREC)
	#define VFACTOR ((1 << SHIFT))
	#define DSIZE 	(256/VFACTOR)
	#define NBLK 	(32<<FPPREC)

	ap_uint<22> batL = B_size;
	ap_uint<22> batd = 0;
	ap_uint<12> id =0;
	ap_uint<8> kd = 0;
	int total_itr = batL*Tiles*M;
	uint256_dt u0 =0, uM =0;
	for(int itr= 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000

		ap_uint<22> bat = batd;
		ap_uint<8> k = kd;
		ap_uint<12> i = id;

		if(i == M -1){
			id = 0;
		} else {
			id++;
		}

		if(i == M -1 && k == Tiles -1){
			kd = 0;
			batd++;
		} else if(i == M -1){
			kd++;
		}

		int ind = k*M+i;
		uint256_dt uI;

		if(i == 0){
			u0 = u_top_in.read();
			uM = u_bottom_in.read();
		}

		uint256_dt vec_a = a_stm_in.read();
		uint256_dt vec_c = c_stm_in.read();
		uint256_dt vec_d = d_stm_in.read();
		for(int v = 0; v < VFACTOR; v++){
			DType a_r = uint2FP_ript<FPPREC, DType>(vec_a.range(DSIZE*(v+1)-1,DSIZE*v));
			DType c_r = uint2FP_ript<FPPREC, DType>(vec_c.range(DSIZE*(v+1)-1,DSIZE*v));
			DType d_r = uint2FP_ript<FPPREC, DType>(vec_d.range(DSIZE*(v+1)-1,DSIZE*v));

			DType u0_r = uint2FP_ript<FPPREC, DType>(u0.range(DSIZE*(v+1)-1,DSIZE*v));
			DType uM_r = uint2FP_ript<FPPREC, DType>(uM.range(DSIZE*(v+1)-1,DSIZE*v));

			DType uI_w = d_r - a_r * u0_r - c_r*uM_r;

			uI.range(DSIZE*(v+1)-1,DSIZE*v) = FP2uint_ript(uI_w);
		}
		int count = bat * Tiles*M + k * M + i;
		if(count < ReadLimit){
			u_out << ((i == 0) ? u0 : (i == M-1) ? uM : uI);
		}
	}

	#undef SHIFT
	#undef VFACTOR
	#undef DSIZE
	#undef NBLK

//	printf("Stage 7 has been returned\n");
}

template <bool FPPREC, class DType, int NMAX>
static void TT_ReducedFW_scalar(hls::stream<float> &ra_stm_in, hls::stream<float> &rb_stm_in, hls::stream<float> &rc_stm_in, hls::stream<float> &rd_stm_in,
		hls::stream<float> &rc_stm_out, hls::stream<float> &rd_stm_out,
		ap_uint<12> B_size, ap_uint<6> Tiles, ap_uint<12> M, unsigned short Sys, ap_uint<8> Ti
		){

	#define SHIFT (3-FPPREC)
	#define VFACTOR ((1 << SHIFT))
	#define DSIZE 	(256/VFACTOR)
	#define NBLK 	(32<<FPPREC)
	const int nblk = NBLK;

	DType  rc_fw[NMAX], rd_fw[NMAX];
	#pragma HLS RESOURCE variable=rc_fw core=XPM_MEMORY uram
	#pragma HLS RESOURCE variable=rd_fw core=XPM_MEMORY uram

	DType vec_b_old_FTM[NBLK], vec_c_old_FTM[NBLK], vec_d_old_FTM[NBLK];

	ap_uint<22> batL = B_size+1;
	ap_uint<22> batd = 0;
	ap_uint<6> id =0;
	ap_uint<7> kd = 0;
	const unsigned char TilesN = (Ti << 1);
	ap_uint<16> TilesN_6 = (ap_uint<16>)Ti * (NBLK<<1);
	int total_itr = B_size*TilesN_6+TilesN_6;
	for(int itr= 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000
		#pragma HLS dependence variable=vec_b_old_FTM RAW distance=n_blk true
		#pragma HLS dependence variable=vec_c_old_FTM RAW distance=n_blk true
		#pragma HLS dependence variable=vec_d_old_FTM RAW distance=n_blk true

		ap_uint<22> bat = batd;
		ap_uint<6> i = id;
		ap_uint<7> k = kd;

		if(i == NBLK -1){
			id = 0;
		} else {
			id++;
		}

		if(i == NBLK -1 && k == TilesN -1){
			kd = 0;
			batd++;
		} else if(i == NBLK -1){
			kd++;
		}

		#pragma HLS dependence variable=vec_b_old_FTM RAW distance=n_blk true
		#pragma HLS dependence variable=vec_c_old_FTM RAW distance=n_blk true
		#pragma HLS dependence variable=vec_d_old_FTM RAW distance=n_blk true

		DType a_r, b_r, c_r, d_r;

		if(i < Sys && bat < B_size){
			a_r = ra_stm_in.read();
			b_r = rb_stm_in.read();
			c_r = rc_stm_in.read();
			d_r = rd_stm_in.read();
		}


		DType b_or = vec_b_old_FTM[i];
		DType c_or = vec_c_old_FTM[i];
		DType d_or = vec_d_old_FTM[i];

		DType denom = (k == 0) ? b_r : b_r - a_r*c_or;
		DType r = 1.0/denom;
		DType c_w1 = c_r;
		DType d_w1 = (k==0) ? d_r : d_r - a_r*d_or;

		DType b_w = 1.0f;
		DType c_w = c_w1*r;
		DType d_w = d_w1*r;


		vec_b_old_FTM[i] = b_w;
		vec_c_old_FTM[i] = c_w;
		vec_d_old_FTM[i] = d_w;


		int offsetW_rr = ((bat & 1) == 0 ? 0 : NMAX/2);
		int offsetR_rr = ((bat & 1) == 1 ? 0 : NMAX/2);

		int pre_add = k + i*TilesN;
		int ind_red =  k + i*TilesN; //(i < Sys) ? pre_add : 255;

		rc_fw[ind_red+offsetW_rr] = c_w; // check this
		rd_fw[ind_red+offsetW_rr] = d_w;


		int ind_red_w = TilesN-1-k + i*TilesN;
		DType tmp_cR = rc_fw[ind_red_w+offsetR_rr];
		DType tmp_dR = rd_fw[ind_red_w+offsetR_rr];
		if(i < Sys && bat > 0){
			rc_stm_out << tmp_cR;
			rd_stm_out << tmp_dR;
		}

	}


	#undef SHIFT
	#undef VFACTOR
	#undef DSIZE
	#undef NBLK

//	printf("Stage 4 has been returned\n");
}


template <bool FPPREC, class DType, int NMAX>
static void TT_ReducedBW_scalar(hls::stream<float> &rc_stm_in, hls::stream<float> &rd_stm_in,
		hls::stream<float> &u_top, hls::stream<float> &u_bottom,
		ap_uint<12> B_size, ap_uint<6> Tiles, ap_uint<12> M, unsigned short Sys, ap_uint<8> Ti
		){

	#define SHIFT (3-FPPREC)
	#define VFACTOR ((1 << SHIFT))
	#define DSIZE 	(256/VFACTOR)
	#define NBLK 	(32<<FPPREC)
	const int nblk = NBLK;

	DType  ru_fw[NMAX];
	#pragma HLS RESOURCE variable=ru_fw core=XPM_MEMORY uram

	DType vec_u_old_RTM[NBLK];
	ap_uint<8> TilesN = (Ti << 1);

	ap_uint<22> batL = B_size+1;
	ap_uint<22> batd = 0;
	ap_uint<6> id =0;
	ap_uint<8> kdd = 0;
	ap_uint<16> TilesN_6 = (ap_uint<16>)Ti * (NBLK<<1);
	int total_itr = B_size*TilesN_6+TilesN_6;
	for(int itr= 0; itr < total_itr; itr++){
		#pragma HLS PIPELINE II=1
		#pragma HLS loop_tripcount min=1638400 max=2000000 avg=2000000
		#pragma HLS dependence variable=vec_u_old_RTM RAW distance=n_blk true

		ap_uint<22> bat = batd;
		ap_uint<6> i = id;
		ap_uint<8> kd = kdd;

		if(i == NBLK -1){
			id = 0;
		} else {
			id++;
		}

		if(i == NBLK -1 && kd == TilesN -1){
			kdd = 0;
			batd++;
		} else if(i == NBLK -1){
			kdd++;
		}

		#pragma HLS dependence variable=vec_u_old_RTM RAW distance=n_blk true
		// forward solve of the reduced diagonal systems
		float c_r, d_r;
		ap_uint<7> k = TilesN -1 -kd;
		int add_ra = k+i*TilesN;
		ap_uint<12> k_3  =(ap_uint<12>)k<<3;
		ap_uint<12> TilesN_3 = ((ap_uint<12>)TilesN<<3);
		int ind_red = (i&7) + (i>>3)*TilesN_3 + k_3;


		if(bat < B_size && i < Sys){
			c_r = rc_stm_in.read();
			d_r = rd_stm_in.read();
		}

		DType u_or = vec_u_old_RTM[i];
		DType f_d = (k == TilesN-1) ? d_r : (d_r-c_r*u_or);
		DType u_w = f_d;

		vec_u_old_RTM[i] = u_w;

		int offsetW_rr = ((bat & 1) == 0 ? 0 : NMAX/2);
		int offsetR_rr = ((bat & 1) == 1 ? 0 : NMAX/2);

		// declare the memory

		ru_fw[ind_red+offsetW_rr] = u_w;



		//				ap_uint<12> add_rev = kd*MAX_Sys+i;
		ap_uint<12> add_rev = kd*NBLK+i;
		ap_uint<12> limit = Sys*TilesN;

		DType tmp_uR = ru_fw[add_rev+offsetR_rr];

		if(((add_rev>>3) & 1) == 0 && bat > 0 && add_rev < limit){
			u_top << tmp_uR;
		} else if(bat > 0 && add_rev < limit){
			u_bottom << tmp_uR;
		}
	}

	#undef SHIFT
	#undef VFACTOR
	#undef DSIZE
	#undef NBLK

//	printf("Stage 6 has been returned\n");
}




#endif
