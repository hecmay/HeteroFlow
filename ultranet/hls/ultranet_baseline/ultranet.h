//===========================================================================
// ultranet.h
//===========================================================================
// @brief: This header file defines the interface for the core functions.

#include <hls_stream.h>
#include <hls_video.h>
#include "stream_tools.h"
#include "config.h"

// image size
#define IN_IMAGE_WIDTH  640
#define IN_IMAGE_HEIGHT 360

#define RESIZE_IMAGE_WIDTH 640
#define RESIZE_IMAGE_HEIGHT 320

#ifdef __SYNTHESIS__
// Top function for synthesis
void ultra_net(
  stream<my_ap_axis >& in,
  stream<my_ap_axis >& out,
  const unsigned int reps
);
#endif

// Top function for cnn accelerator
void do_compute(
// #ifndef __SYNTHESIS__
//     ap_uint<CONV_0_IN_BIT> before_resize[IN_IMAGE_HEIGHT][IN_IMAGE_WIDTH][3],
//     ap_uint<CONV_0_IN_BIT> after_resize[RESIZE_IMAGE_HEIGHT][RESIZE_IMAGE_WIDTH][3],
//     ap_uint<CONV_0_OUT_BIT> conv0_out[CONV_0_OFM_ROW][CONV_0_OFM_COL][CONV_0_OFM_CH],
//     ap_uint<CONV_0_OUT_BIT> pool0_out[CONV_1_IFM_ROW][CONV_1_IFM_COL][CONV_1_IFM_CH],
//     ap_uint<CONV_1_OUT_BIT> conv1_out[CONV_1_OFM_ROW][CONV_1_OFM_COL][CONV_1_OFM_CH],
//     ap_uint<CONV_1_OUT_BIT> pool1_out[CONV_2_IFM_ROW][CONV_2_IFM_COL][CONV_2_IFM_CH],
//     ap_uint<CONV_2_OUT_BIT> conv2_out[CONV_2_OFM_ROW][CONV_2_OFM_COL][CONV_2_OFM_CH],
//     ap_uint<CONV_2_OUT_BIT> pool2_out[CONV_3_IFM_ROW][CONV_3_IFM_COL][CONV_3_IFM_CH],
//     ap_uint<CONV_3_OUT_BIT> conv3_out[CONV_3_OFM_ROW][CONV_3_OFM_COL][CONV_3_OFM_CH],
//     ap_uint<CONV_3_OUT_BIT> pool3_out[CONV_4_IFM_ROW][CONV_4_IFM_COL][CONV_4_IFM_CH],
//     ap_uint<CONV_4_OUT_BIT> conv4_out[CONV_4_OFM_ROW][CONV_4_OFM_COL][CONV_4_OFM_CH],
//     ap_uint<CONV_5_OUT_BIT> conv5_out[CONV_5_OFM_ROW][CONV_5_OFM_COL][CONV_5_OFM_CH],
//     ap_uint<CONV_6_OUT_BIT> conv6_out[CONV_6_OFM_ROW][CONV_6_OFM_COL][CONV_6_OFM_CH],
//     ap_uint<CONV_7_OUT_BIT> conv7_out[CONV_7_OFM_ROW][CONV_7_OFM_COL][CONV_7_OFM_CH],
// #endif
    stream<my_ap_axis >& in,
    stream<my_ap_axis >& out,
    const unsigned int reps
);

#if defined(DEBUG) || defined(TEST)
void load_data(const char *path, char *ptr, unsigned int size);
void write_data(const char *path, char *ptr, unsigned int size);
#endif