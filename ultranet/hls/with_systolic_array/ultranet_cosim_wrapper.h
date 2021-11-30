#include <hls_stream.h>
#include "stream_tools.h"

void ultranet_cosim_wrapper(
    uint8_t  test_img[360 * 640 * 3],
    my_ap_axis  hls_results[512],
    unsigned out_len[1]
);