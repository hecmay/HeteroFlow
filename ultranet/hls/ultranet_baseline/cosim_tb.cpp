//=========================================================================
// cosim_tb.cpp
//=========================================================================
// @brief: HLS cosim testbench for Ultranet CNN accelerator

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include "ultranet_cosim_wrapper.h"
#include <hls_video.h>

using namespace std;

//------------------------------------------------------------------------
// Correctness Check
//------------------------------------------------------------------------

int main(int argc, char** argv){
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <mode>" << endl;
        cout << "\tmode:" << endl
             << "\t  presim: run C-simulation and record outputs" << endl
             << "\t  postsim: run RTL-simulation and record outputs" << endl;
        cout << "Aborting..." << endl;
    }
    string mode = argv[1];
    string output_record_file;
    if (mode == "presim") {
        output_record_file = "../../../presim_out.dat";
    } else if (mode == "postsim") {
        output_record_file = "./postsim_out.dat";
    } else {
        cout << "ERROR! unknown mode option:" << mode << endl;
        return 1;
    }

    // generate a test image
    uint8_t  test_img[360 * 640 * 3];
    for (size_t i = 0; i < 360 * 640 * 3; i++) {
        test_img[i] = (uint8_t)(rand() % 256);
    }

    // space for output
    my_ap_axis results[3650];
    unsigned len[1];

    // invoke test wraper
    ultranet_cosim_wrapper(test_img, results, len);

    // store results
    ofstream f(output_record_file);
    if (!f) {
        cout << "ERROR when opening/creating output file!" << endl;
        return 1;
    }
    f << "total length = " << len[0] << endl;
    for (size_t i = 0; i < len[0]; i++) {
        f << dec << i << ": ";
        f << "0x" << hex << results[i].data << endl;
    }
    f.close();

    return 0;
}
