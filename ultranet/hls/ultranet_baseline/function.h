#pragma once
#include <hls_stream.h>
#include <ap_int.h>
using namespace std;
#include <assert.h>
#include "stream_tools.h"

/**
 *  padding 函数
 *  数据宽度为 IN_BIT * SIMD
 * 
 */ 
template <	unsigned IN_BIT, 
			unsigned SIMD,
			unsigned P>
void padding_var(
    // 将每一数竖看成一个元素
	stream<ap_uint<IN_BIT * SIMD> >& in,
	stream<ap_uint<IN_BIT * SIMD> >& out,
	const unsigned in_row,				// 
	const unsigned in_col,				// 
	const unsigned in_simd_pre_ch,		// ch / simd
	const unsigned reps = 1)
{
    const unsigned OUT_COL = in_col + 2 * P;

	for (unsigned rep = 0; rep < reps; rep++) {
		#pragma HLS loop_tripcount min=1 max=1
		for (unsigned h = 0; h < P; h++) {
			#pragma HLS loop_tripcount min=1 max=1
			for (unsigned s = 0; s < OUT_COL; s++) {
				#pragma HLS loop_tripcount min=40 max=640
				// 将一 ch 的数据置零
				append_zero<IN_BIT * SIMD>(out, in_simd_pre_ch);
			}
		}

		for (unsigned h = 0; h < in_row; h++) {
			#pragma HLS loop_tripcount min=20 max=320
			for ( unsigned s = 0; s < OUT_COL; s++ ) {
// #pragma HLS PIPELINE II=1
				#pragma HLS loop_tripcount min=3 max=36
				if ( (s < P) || (s >= OUT_COL-P) ) {
					// temp_out = 0;
					append_zero<IN_BIT * SIMD>(out, in_simd_pre_ch);
				}
				else {
					// cout << "in size :" << in.size() << endl;
					stream_move<IN_BIT * SIMD>(in, out, in_simd_pre_ch);

				}
				// out.write(temp_out);
			}
		}

		for (unsigned h = 0; h < P; h++) {
			#pragma HLS loop_tripcount min=1 max=1
			for (unsigned i = 0; i < OUT_COL; i++) {
				#pragma HLS loop_tripcount min=3 max=36
				append_zero<IN_BIT * SIMD>(out, in_simd_pre_ch);
			}
		}

	}
}

/**
 *  padding 函数
 */ 
template <	unsigned IN_ROW,
			unsigned IN_COL,
            unsigned IN_CH,
			unsigned IN_BIT, 
			unsigned P>
void padding(
    // 将每一数竖看成一个元素
	stream<ap_uint<IN_CH*IN_BIT> >& in,
	stream<ap_uint<IN_CH*IN_BIT> >& out,
	const unsigned reps = 1)
{
    const unsigned OUT_ROW = IN_ROW + 2 * P;
    const unsigned OUT_COL = IN_COL + 2 * P;

	ap_uint<IN_CH*IN_BIT> temp_out = 0;

	for (unsigned rep = 0; rep < reps; rep++) {
		#pragma HLS loop_tripcount min=1 max=1
		for (unsigned h = 0; h < P; h++) {
			#pragma HLS loop_tripcount min=1 max=1
			for (unsigned s = 0; s < OUT_COL; s++) {
				#pragma HLS loop_tripcount min=3 max=36
				out.write(0);
			}
		}

		for (unsigned h = 0; h < IN_ROW; h++) {
			#pragma HLS loop_tripcount min=20 max=320
			for ( unsigned s = 0; s < OUT_COL; s++ ) {
				#pragma HLS PIPELINE II=1
				#pragma HLS loop_tripcount min=20 max=320

				if ( (s < P) || (s >= OUT_COL-P) ) {
					temp_out = 0;
				}
				else {
					temp_out = in.read();
				}
				
				out.write(temp_out);
			}
		}

		for (unsigned h = 0; h < P; h++) {
			#pragma HLS loop_tripcount min=1 max=1
			for (unsigned i = 0; i < OUT_COL; i++) {
				#pragma HLS loop_tripcount min=20 max=320
				out.write(0);
			}
		}

	}
}

template <	unsigned IN_BIT,
			unsigned OUT_BIT,
			unsigned INC_BIT,
			unsigned BIAS_BIT,

			unsigned DATA_BIT,
			unsigned W_BIT,
			unsigned L_SHIFT
			>
ap_uint<OUT_BIT> bn_qurelu( ap_int<IN_BIT> in,
                ap_int<INC_BIT> inc,
                ap_int<BIAS_BIT> bias ) {   

	const unsigned D = 1 << (W_BIT - 1 + DATA_BIT + L_SHIFT);

	ap_int<IN_BIT> bn_res = in * inc + bias;
	ap_uint<OUT_BIT> res;

	if (bn_res > 0) {
		bn_res = (bn_res + (D >> 1)) >> (W_BIT - 1 + DATA_BIT + L_SHIFT);
		if (bn_res > 15){
			res = 15;
		} else {
			res = bn_res;
		}
	} else {
		res = 0;
	}
	return res;
    
}