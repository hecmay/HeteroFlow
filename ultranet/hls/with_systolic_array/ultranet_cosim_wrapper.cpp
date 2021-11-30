// @brief: cosim wrapper for ultranet CNN xcel
#include "ultranet.h"

// dump input image into the input stream
const int nums_line_pre_img = 360 * 640 * 3 * 8/ 64;
const int data_points_per_line = 8;
void data_feeder(
    uint8_t  test_img[360 * 640 * 3],
    hls::stream<my_ap_axis> &input_stream
) {
    for (unsigned int i = 0; i < nums_line_pre_img; i++) {
        my_ap_axis temp;
        #pragma HLS pipeline II=1
        for (unsigned int j = 0; j < data_points_per_line; j++) {
            #pragma HLS loop_flatten
            temp.data( 8*(j+1)-1, 8*j ) = test_img[i * data_points_per_line + j];
        }
        input_stream.write(temp);
    }
}

// receive output from the output stream
void data_collector(
    hls::stream<my_ap_axis> &output_stream,
    my_ap_axis  hls_results[3650], // maybe too small?
    unsigned out_len[1]
) {
    bool exit = false;
    unsigned cnt = 0;
    while (!exit) {
        #pragma HLS pipeline II=1
        my_ap_axis in = output_stream.read();
        exit = (in.last == 1);
        hls_results[cnt] = in;
        cnt++;
    }
    out_len[0] = cnt;
}

void ultranet_cosim_wrapper(
    uint8_t  test_img[360 * 640 * 3],
    my_ap_axis  hls_results[3650],
    unsigned out_len[1]
){
    #pragma HLS dataflow
    hls::stream<my_ap_axis> input_stream("input stream");
    hls::stream<my_ap_axis> output_stream("output stream");
    #pragma HLS stream variable=input_stream depth=1024
    #pragma HLS stream variable=output_stream depth=1024

    data_feeder(test_img, input_stream);
    do_compute(input_stream, output_stream, 1);
    data_collector(output_stream, hls_results, out_len);
}