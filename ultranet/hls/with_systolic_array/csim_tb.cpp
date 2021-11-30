//=========================================================================
// csim_tb.cpp
//=========================================================================
// @brief: HLS csim testbench for Ultranet CNN accelerator

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cstdlib>
#include <ap_int.h>
#include <hls_stream.h>
#include "ultranet.h"

using namespace std;

const unsigned IMG_H = 360;
const unsigned IMG_W = 640;
const unsigned GRID_H = 20;
const unsigned GRID_W = 40;
const unsigned IN_CH = 3;
const unsigned OUT_CH = 36;

const unsigned IN_SIZE = IMG_H * IMG_W * IN_CH;
const unsigned OUT_SIZE = GRID_H * GRID_W * OUT_CH;

const int IMG_BW = 8;
const int IN_BW_ACQ = 8;
const int DMA_BW = 64;

//------------------------------------------------------------------------
// Correctness Check
//------------------------------------------------------------------------

// dump input image into the input stream

void data_feeder(
    uint8_t test_img[IN_SIZE],
    hls::stream<my_ap_axis> &input_stream
) {
    // cout << "Tracing --\ndata_feeder(\n";
    // cout << "\ttest_img @ " << hex << (void*)&(test_img[0]) << "\n";
    // cout << "\tout @ " << hex << (void*)&input_stream << "\n";
    // cout << "\tIN_SIZE = " << dec << IN_SIZE << "\n";
    // cout << ");\n";
    unsigned pack_cnt = 0;
    my_ap_axis temp;
    for (size_t row = 0; row < IMG_H; row++) {
        for (size_t col = 0; col < IMG_W; col++) {
            for (size_t ch = 0; ch < IN_CH; ch++)  {
                uint8_t pixel_component = test_img[ch + col * IN_CH + row * IMG_W * IN_CH];
                // ap_uint<IN_BW_ACQ> pacq = (pixel_component & 0xF0) >> 4;
                ap_uint<IN_BW_ACQ> pacq = pixel_component;
                temp.data(IN_BW_ACQ * (pack_cnt + 1) - 1, IN_BW_ACQ * pack_cnt) = pacq;
                pack_cnt++;
                if (pack_cnt == DMA_BW / IN_BW_ACQ) {
                    pack_cnt = 0;
                    input_stream.write(temp);
                }
            }
        }
    }
}

// receive output from the output stream
void update_counter(int &row, int &col, int &ch) {
    ch++;
    if (ch == OUT_CH) {
        ch = 0;
        col ++;
        if (col == GRID_W) {
            col = 0;
            row ++;
            if (row == GRID_H) {
                row = 0;
            }
        }
    }
}
void data_collector(
    hls::stream<my_ap_axis> &output_stream,
    int32_t hls_results[GRID_H][GRID_W][OUT_CH]
) {
    bool last = false;
    int ch = 0;
    int col = 0;
    int row = 0;
    size_t cnt = 0;
    while (!last) {
        if (output_stream.empty()) {
            cout << "Error! output stream empty in C-sim!" << endl;
            exit(EXIT_FAILURE);
        }
        if (cnt > OUT_SIZE) {
            cout << "Error! Output size exceed limit ("  << cnt << " > " << OUT_SIZE << ")!" << endl;
            exit(EXIT_FAILURE);
        }

        my_ap_axis in = output_stream.read();
        last = (in.last == 1);
        hls_results[row][col][ch] = in.data(31, 0);
        cnt++;
        update_counter(row, col, ch);
        hls_results[row][col][ch] = in.data(63, 32);
        update_counter(row, col, ch);
        cnt++;
    }
}

void ultranet_csim_wrapper(
    uint8_t test_img[IN_SIZE],
    int32_t hls_results[GRID_H][GRID_W][OUT_CH],
    ap_uint<CONV_0_IN_BIT> before_resize[IN_IMAGE_HEIGHT][IN_IMAGE_WIDTH][3],
    ap_uint<CONV_0_IN_BIT> after_resize[RESIZE_IMAGE_HEIGHT][RESIZE_IMAGE_WIDTH][3],
    ap_uint<CONV_0_OUT_BIT> conv0_out[CONV_0_OFM_ROW][CONV_0_OFM_COL][CONV_0_OFM_CH],
    ap_uint<CONV_0_OUT_BIT> pool0_out[CONV_1_IFM_ROW][CONV_1_IFM_COL][CONV_1_IFM_CH],
    ap_uint<CONV_1_OUT_BIT> conv1_out[CONV_1_OFM_ROW][CONV_1_OFM_COL][CONV_1_OFM_CH],
    ap_uint<CONV_1_OUT_BIT> pool1_out[CONV_2_IFM_ROW][CONV_2_IFM_COL][CONV_2_IFM_CH],
    ap_uint<CONV_2_OUT_BIT> conv2_out[CONV_2_OFM_ROW][CONV_2_OFM_COL][CONV_2_OFM_CH],
    ap_uint<CONV_2_OUT_BIT> pool2_out[CONV_3_IFM_ROW][CONV_3_IFM_COL][CONV_3_IFM_CH],
    ap_uint<CONV_3_OUT_BIT> conv3_out[CONV_3_OFM_ROW][CONV_3_OFM_COL][CONV_3_OFM_CH],
    ap_uint<CONV_3_OUT_BIT> pool3_out[CONV_4_IFM_ROW][CONV_4_IFM_COL][CONV_4_IFM_CH],
    ap_uint<CONV_4_OUT_BIT> conv4_out[CONV_4_OFM_ROW][CONV_4_OFM_COL][CONV_4_OFM_CH],
    ap_uint<CONV_5_OUT_BIT> conv5_out[CONV_5_OFM_ROW][CONV_5_OFM_COL][CONV_5_OFM_CH],
    ap_uint<CONV_6_OUT_BIT> conv6_out[CONV_6_OFM_ROW][CONV_6_OFM_COL][CONV_6_OFM_CH],
    ap_uint<CONV_7_OUT_BIT> conv7_out[CONV_7_OFM_ROW][CONV_7_OFM_COL][CONV_7_OFM_CH]
){
    hls::stream<my_ap_axis> input_stream("input stream");
    hls::stream<my_ap_axis> output_stream("output stream");

    data_feeder(test_img, input_stream);
    cout << "Successfully feed data into xcel" << endl;
    do_compute(
        before_resize,
        after_resize,
        conv0_out,
        pool0_out,
        conv1_out,
        pool1_out,
        conv2_out,
        pool2_out,
        conv3_out,
        pool3_out,
        conv4_out,
        conv5_out,
        conv6_out,
        conv7_out,
        input_stream, output_stream, 1);
    cout << "xcel complete!" << endl;
    data_collector(output_stream, hls_results);
    cout << "Successfully collected results from xcel" << endl;
}


// store results
template<unsigned H, unsigned W, unsigned C, unsigned B>
void save_results(
    ap_uint<B> data[H][W][C],
    string filename
) {
    ofstream out_result_handle(filename);
    if (!out_result_handle) {
        cout << "ERROR when opening/creating output file: " << filename << endl;
        exit(EXIT_FAILURE);
    }
    for (size_t row = 0; row < H; row++)  {
        for (size_t col = 0; col < W; col++) {
            for (size_t ch = 0; ch < C; ch++) {
                out_result_handle << data[row][col][ch] << endl;
            }
        }
    }

    out_result_handle.close();
}

template<unsigned H, unsigned W, unsigned C, unsigned B>
void save_results(
    int32_t data[H][W][C],
    string filename
) {
    ofstream out_result_handle(filename);
    if (!out_result_handle) {
        cout << "ERROR when opening/creating output file: " << filename << endl;
        exit(EXIT_FAILURE);
    }
    for (size_t row = 0; row < H; row++)  {
        for (size_t col = 0; col < W; col++) {
            for (size_t ch = 0; ch < C; ch++) {
                out_result_handle << dec << data[row][col][ch] << endl;
            }
        }
    }

    out_result_handle.close();
}

// space for output
// declare them here to prevent stack overflow
int32_t results[GRID_H][GRID_W][OUT_CH];
ap_uint<CONV_0_IN_BIT> before_resize[IN_IMAGE_HEIGHT][IN_IMAGE_WIDTH][3];
ap_uint<CONV_0_IN_BIT> after_resize[RESIZE_IMAGE_HEIGHT][RESIZE_IMAGE_WIDTH][3];
ap_uint<CONV_0_OUT_BIT> conv0_out[CONV_0_OFM_ROW][CONV_0_OFM_COL][CONV_0_OFM_CH];
ap_uint<CONV_0_OUT_BIT> pool0_out[CONV_1_IFM_ROW][CONV_1_IFM_COL][CONV_1_IFM_CH];
ap_uint<CONV_1_OUT_BIT> conv1_out[CONV_1_OFM_ROW][CONV_1_OFM_COL][CONV_1_OFM_CH];
ap_uint<CONV_1_OUT_BIT> pool1_out[CONV_2_IFM_ROW][CONV_2_IFM_COL][CONV_2_IFM_CH];
ap_uint<CONV_2_OUT_BIT> conv2_out[CONV_2_OFM_ROW][CONV_2_OFM_COL][CONV_2_OFM_CH];
ap_uint<CONV_2_OUT_BIT> pool2_out[CONV_3_IFM_ROW][CONV_3_IFM_COL][CONV_3_IFM_CH];
ap_uint<CONV_3_OUT_BIT> conv3_out[CONV_3_OFM_ROW][CONV_3_OFM_COL][CONV_3_OFM_CH];
ap_uint<CONV_3_OUT_BIT> pool3_out[CONV_4_IFM_ROW][CONV_4_IFM_COL][CONV_4_IFM_CH];
ap_uint<CONV_4_OUT_BIT> conv4_out[CONV_4_OFM_ROW][CONV_4_OFM_COL][CONV_4_OFM_CH];
ap_uint<CONV_5_OUT_BIT> conv5_out[CONV_5_OFM_ROW][CONV_5_OFM_COL][CONV_5_OFM_CH];
ap_uint<CONV_6_OUT_BIT> conv6_out[CONV_6_OFM_ROW][CONV_6_OFM_COL][CONV_6_OFM_CH];
ap_uint<CONV_7_OUT_BIT> conv7_out[CONV_7_OFM_ROW][CONV_7_OFM_COL][CONV_7_OFM_CH];

int main(int argc, char** argv){
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <img_path> <output_path>" << endl;
        cout << "Aborting..." << endl;
        exit(EXIT_FAILURE);
    }
    string img_path = argv[1];
    string output_dir = argv[2];

    // load test image
    uint8_t test_img[IN_SIZE];
    ifstream in_img_handle(img_path);
    if (!in_img_handle) {
        cout << "ERROR when opening input file: " << img_path << endl;
        exit(EXIT_FAILURE);
    }
    string line;
    int ln = 0;
    // cout << "Tracing --\nload test image in main(\n";
    // cout << "\ttest_img @ " << hex << (void*)&(test_img[0]) << "\n";
    // cout << "\tin_img_handle @ " << hex << (void*)&in_img_handle << "\n";
    // cout << "\tIN_SIZE = " << dec << IN_SIZE << "\n";
    // cout << ");\n";
    while (getline(in_img_handle, line, '\n')) {
        int val = stoi(line);

        // cout << dec << ln << ": " << line;
        // cout << " => " << hex << val << "\n";

        test_img[ln] = val;
        ln++;
    }
    in_img_handle.close();
    if (ln != IN_SIZE) {
        cout << "ERROR loading input image: size mismatch " << endl;
        cout << "\tExpected:" << IN_SIZE << endl;
        cout << "\tActual  :" << ln << endl;
        exit(EXIT_FAILURE);
    }
    cout << "Input image loading success!" << endl;

    // invoke test wraper
    ultranet_csim_wrapper(
        test_img, results,
        before_resize,
        after_resize,
        conv0_out,
        pool0_out,
        conv1_out,
        pool1_out,
        conv2_out,
        pool2_out,
        conv3_out,
        pool3_out,
        conv4_out,
        conv5_out,
        conv6_out,
        conv7_out
    );

    save_results<IN_IMAGE_HEIGHT, IN_IMAGE_WIDTH, 3, CONV_0_IN_BIT>(
        before_resize, output_dir + "/xcel_before_resize.dat");
    save_results<RESIZE_IMAGE_HEIGHT, RESIZE_IMAGE_WIDTH, 3, CONV_0_IN_BIT>(
        after_resize, output_dir + "/xcel_after_resize.dat");

    save_results<CONV_0_OFM_ROW, CONV_0_OFM_COL, CONV_0_OFM_CH, CONV_0_OUT_BIT>(
        conv0_out, output_dir + "/xcel_conv0.dat");
    save_results<CONV_1_OFM_ROW, CONV_1_OFM_COL, CONV_1_OFM_CH, CONV_1_OUT_BIT>(
        conv1_out, output_dir + "/xcel_conv1.dat");
    save_results<CONV_2_OFM_ROW, CONV_2_OFM_COL, CONV_2_OFM_CH, CONV_2_OUT_BIT>(
        conv2_out, output_dir + "/xcel_conv2.dat");
    save_results<CONV_3_OFM_ROW, CONV_3_OFM_COL, CONV_3_OFM_CH, CONV_3_OUT_BIT>(
        conv3_out, output_dir + "/xcel_conv3.dat");
    save_results<CONV_4_OFM_ROW, CONV_4_OFM_COL, CONV_4_OFM_CH, CONV_4_OUT_BIT>(
        conv4_out, output_dir + "/xcel_conv4.dat");
    save_results<CONV_5_OFM_ROW, CONV_5_OFM_COL, CONV_5_OFM_CH, CONV_5_OUT_BIT>(
        conv5_out, output_dir + "/xcel_conv5.dat");
    save_results<CONV_6_OFM_ROW, CONV_6_OFM_COL, CONV_6_OFM_CH, CONV_6_OUT_BIT>(
        conv6_out, output_dir + "/xcel_conv6.dat");
    save_results<CONV_7_OFM_ROW, CONV_7_OFM_COL, CONV_7_OFM_CH, CONV_7_OUT_BIT>(
        conv7_out, output_dir + "/xcel_conv7.dat");
    save_results<GRID_H, GRID_W, OUT_CH, 32>(
        results, output_dir + "/xcel_result.dat");

    save_results<CONV_1_IFM_ROW, CONV_1_IFM_COL, CONV_1_IFM_CH, CONV_0_OUT_BIT>(
        pool0_out, output_dir + "/xcel_pool0.dat");
    save_results<CONV_2_IFM_ROW, CONV_2_IFM_COL, CONV_2_IFM_CH, CONV_1_OUT_BIT>(
        pool1_out, output_dir + "/xcel_pool1.dat");
    save_results<CONV_3_IFM_ROW, CONV_3_IFM_COL, CONV_3_IFM_CH, CONV_2_OUT_BIT>(
        pool2_out, output_dir + "/xcel_pool2.dat");
    save_results<CONV_4_IFM_ROW, CONV_4_IFM_COL, CONV_4_IFM_CH, CONV_3_OUT_BIT>(
        pool3_out, output_dir + "/xcel_pool3.dat");

    return 0;
}