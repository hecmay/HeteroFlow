#include "ihc_apint.h"
#pragma OPENCL EXTENSION cl_intel_arbitrary_precision_integers : enable
#pragma OPENCL EXTENSION cl_intel_channels : enable
channel int32_t c_buf_1;
channel int32_t c_buf_2;
channel int32_t c_buf_3;
channel int32_t c_buf_4;
channel int32_t c_buf_5;
channel int32_t c_buf_6;
channel int32_t c_buf_7;
channel int32_t c_buf_8;
channel int32_t c_buf_9;

__kernel void flow_calc(__global float* restrict flow_calc_output) {
    for (int32_t r = 0; r < 436; ++r) {
      for (int32_t c = 0; c < 1024; ++c) {
        if ((((2 <= r) && (r < 434)) && (2 <= c)) && (c < 1022)) {
          int32_t scalar0;
          scalar0 = (read_channel_intel(c_buf_9));
          float a;
          int8_t _converter;
          _converter = ((uint8_t)((scalar0 >> 8) & ((1L << (0 - 8)) - 1)));
          a = ((float)_converter);
          float b;
          union { uint32_t from; float to;} _converter1;
          _converter1.from = ((uint32_t)((scalar0 >> 40) & ((1L << (8 - 40)) - 1)));
          b = _converter1.to;
          int32_t c1;
          union { uint32_t from; float to;} _converter2;
          _converter2.from = ((uint32_t)((scalar0 >> 72) & ((1L << (40 - 72)) - 1)));
          c1 = _converter2.to;
          float d;
          int8_t _converter3;
          _converter3 = ((uint8_t)((scalar0 >> 80) & ((1L << (72 - 80)) - 1)));
          d = ((float)_converter3);
          float e;
          union { uint32_t from; float to;} _converter4;
          _converter4.from = ((uint32_t)((scalar0 >> 112) & ((1L << (80 - 112)) - 1)));
          e = _converter4.to;
          float f;
          union { uint32_t from; float to;} _converter5;
          _converter5.from = ((uint32_t)((scalar0 >> 144) & ((1L << (112 - 144)) - 1)));
          f = _converter5.to;
          float denom;
          denom = ((a * b) - (d * d));
          union { float from; uint32_t to;} _converter6;
          _converter6.from = ((e * (d - b)) / denom);
          flow_calc_output[(c1 + ((r * 1024)))] = _converter6.to;
          union { float from; uint32_t to;} _converter7;
          _converter7.from = ((e * (d - a)) / denom);
          flow_calc_output[(c1 + ((r * 1024)))] = _converter7.to;
        }
      }
    }
}

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void tensor_weight_x() {
    float tensor_weight_x_tensor_y_reuse[3 * 1024];
    float t_w[3];
    t_w[0] = 3.243000e-01f;
    t_w[1] = 3.513000e-01f;
    t_w[2] = 3.243000e-01f;
    for (int32_t y_reuse = 0; y_reuse < 438; ++y_reuse) {
      for (int32_t x = 0; x < 1024; ++x) {
        for (int32_t tensor_weight_x_tensor_y_1 = 0; tensor_weight_x_tensor_y_1 < 2; ++tensor_weight_x_tensor_y_1) {
          tensor_weight_x_tensor_y_reuse[(x + (tensor_weight_x_tensor_y_1 * 1024))] = tensor_weight_x_tensor_y_reuse[((x + (tensor_weight_x_tensor_y_1 * 1024)) + 1024)];
        }
        tensor_weight_x_tensor_y_reuse[(x + 2048)] = read_channel_intel(c_buf_8);
        if (2 <= y_reuse) {
          float reducer5;
          reducer5 = 0.000000e+00f;
          for (int32_t rdx_x = 0; rdx_x < 3; ++rdx_x) {
            reducer5 = ((tensor_weight_x_tensor_y_reuse[((x + (rdx_x * 1024)) + -1024)] * t_w[rdx_x]) + reducer5);
          }
          write_channel_intel(c_buf_9, (float)((x < 1) ? 0.000000e+00f : reducer5));
        }
      }
    }
}

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void tensor_weight_y() {
    float tensor_weight_y_out_product_reuse[3*1024];
    float t_w[3];
    t_w[0] = 3.243000e-01f;
    t_w[1] = 3.513000e-01f;
    t_w[2] = 3.243000e-01f;
    for (int32_t y_reuse = 0; y_reuse < 438; ++y_reuse) {
      for (int32_t x = 0; x < 1024; ++x) {
        for (int32_t tensor_weight_y_out_product_1 = 0; tensor_weight_y_out_product_1 < 2; ++tensor_weight_y_out_product_1) {
          tensor_weight_y_out_product_reuse[(x + (tensor_weight_y_out_product_1 * 1024))] = tensor_weight_y_out_product_reuse[((x + (tensor_weight_y_out_product_1 * 1024)) + 1024)];
        }
        tensor_weight_y_out_product_reuse[(x + 2048)] = read_channel_intel(c_buf_7);
        if (2 <= y_reuse) {
          float reducer4;
          reducer4 = 0.000000e+00f;
          for (int32_t rdx_y = 0; rdx_y < 3; ++rdx_y) {
            reducer4 = ((tensor_weight_y_out_product_reuse[((x + (rdx_y * 1024)) + -1024)] * t_w[rdx_y]) + reducer4);
          }
          write_channel_intel(c_buf_8, (float)(((3 <= y_reuse) && (y_reuse < 437)) ? reducer4 : 0.000000e+00f));
        }
      }
    }
}

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void outer_product() {
    for (int32_t y = 0; y < 436; ++y) {
      for (int32_t x = 0; x < 1024; ++x) {
        ap_uint<144> temp;
        int32_t t;
        t = read_channel_intel(c_buf_6);
        float a;
        int32_t _converter;
        _converter = ((uint32_t)((t >> 32) & ((1L << (0 - 32)) - 1)));
        a = ((float)_converter);
        float b;
        union { uint32_t from; float to;} _converter1;
        _converter1.from = ((uint32_t)((t >> 64) & ((1L << (32 - 64)) - 1)));
        b = _converter1.to;
        float c;
        union { uint32_t from; float to;} _converter2;
        _converter2.from = ((uint32_t)((t >> 96) & ((1L << (64 - 96)) - 1)));
        c = _converter2.to;
        union { float from; uint32_t to;} _converter3;
        _converter3.from = (a * a);
        temp += _converter3.to;
        union { float from; uint32_t to;} _converter4;
        _converter4.from = (b * b);
        temp += _converter4.to;
        union { float from; uint32_t to;} _converter5;
        _converter5.from = (c * c);
        temp += _converter5.to;
        union { float from; uint32_t to;} _converter6;
        _converter6.from = (a * b);
        temp += _converter6.to;
        union { float from; uint32_t to;} _converter7;
        _converter7.from = (a * c);
        temp += _converter7.to;
        union { float from; uint32_t to;} _converter8;
        _converter8.from = (b * c);
        temp += _converter8.to;
        write_channel_intel(c_buf_7, temp);
      }
    }
}

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void grad_weight_x() {
    float grad_weight_x_y_filt_reuse[1*7];
    float g_f[7];
    g_f[0] = 7.550000e-02f;
    g_f[1] = 1.330000e-01f;
    g_f[2] = 1.869000e-01f;
    g_f[3] = 2.903000e-01f;
    g_f[4] = 1.869000e-01f;
    g_f[5] = 1.330000e-01f;
    g_f[6] = 7.550000e-02f;
    for (int32_t y = 0; y < 436; ++y) {
      for (int32_t x_reuse = 0; x_reuse < 1024; ++x_reuse) {
        for (int32_t grad_weight_x_y_filt_0 = 0; grad_weight_x_y_filt_0 < 6; ++grad_weight_x_y_filt_0) {
          grad_weight_x_y_filt_reuse[grad_weight_x_y_filt_0] = grad_weight_x_y_filt_reuse[(grad_weight_x_y_filt_0 + 1)];
        }
        grad_weight_x_y_filt_reuse[6] = read_channel_intel(c_buf_5);
        if (6 <= x_reuse) {
          float reducer3;
          reducer3 = 0.000000e+00f;
          for (int32_t rd = 0; rd < 7; ++rd) {
            reducer3 = ((grad_weight_x_y_filt_reuse[rd] * g_f[rd]) + reducer3);
          }
          write_channel_intel(c_buf_6, reducer3);
        }
      }
    }
}

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void grad_weight_y() {
    float grad_weight_y_pack_reuse[7*1024];
    float g_f[7];
    g_f[0] = 7.550000e-02f;
    g_f[1] = 1.330000e-01f;
    g_f[2] = 1.869000e-01f;
    g_f[3] = 2.903000e-01f;
    g_f[4] = 1.869000e-01f;
    g_f[5] = 1.330000e-01f;
    g_f[6] = 7.550000e-02f;
    for (int32_t y_reuse = 0; y_reuse < 436; ++y_reuse) {
      for (int32_t x = 0; x < 1024; ++x) {
        for (int32_t grad_weight_y_pack_1 = 0; grad_weight_y_pack_1 < 6; ++grad_weight_y_pack_1) {
          grad_weight_y_pack_reuse[(x + (grad_weight_y_pack_1 * 1024))] = grad_weight_y_pack_reuse[((x + (grad_weight_y_pack_1 * 1024)) + 1024)];
        }
        grad_weight_y_pack_reuse[(x + 6144)] = read_channel_intel(c_buf_4);
        if (6 <= y_reuse) {
          float reducer2;
          reducer2 = 0.000000e+00f;
          for (int32_t rdx = 0; rdx < 7; ++rdx) {
            reducer2 = ((grad_weight_y_pack_reuse[(x + (rdx * 1024))] * g_f[rdx]) + reducer2);
          }
          write_channel_intel(c_buf_5, reducer2);
        }
      }
    }
}

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void grad_pack() {
    for (int32_t y = 0; y < 436; ++y) {
      for (int32_t x = 0; x < 1024; ++x) {
        float t;
        t = 0.000000e+00f;
        union { float from; uint32_t to;} _converter;
        _converter.from = read_channel_intel(c_buf_1);
        t += _converter.to;
        union { float from; uint32_t to;} _converter1;
        _converter1.from = read_channel_intel(c_buf_2);
        t += _converter1.to;
        union { float from; uint32_t to;} _converter2;
        _converter2.from = read_channel_intel(c_buf_3);
        t += _converter2.to;
        write_channel_intel(c_buf_4, t);
      }
    }
}

__kernel void calc_z_gradient(__global float* restrict calc_z_gradient_img0, __global float* restrict calc_z_gradient_img1, __global float* restrict calc_z_gradient_img2, __global float* restrict calc_z_gradient_img3, __global float* restrict calc_z_gradient_img4) {
    int32_t g_w[5];
    g_w[0] = 1;
    g_w[1] = -8;
    g_w[2] = 0;
    g_w[3] = 8;
    g_w[4] = 1;
    for (int32_t y = 0; y < 436; ++y) {
      for (int32_t x = 0; x < 1024; ++x) {
        write_channel_intel(c_buf_3, ((((((calc_z_gradient_img0[(x + (y * 1024))] * ((float)g_w[0])) + (calc_z_gradient_img1[(x + (y * 1024))] * ((float)g_w[1]))) + (calc_z_gradient_img2[(x + (y * 1024))] * ((float)g_w[2]))) + (calc_z_gradient_img3[(x + (y * 1024))] * ((float)g_w[3]))) + (calc_z_gradient_img4[(x + (y * 1024))] * ((float)g_w[4]))) * 8.333334e-02f));
      }
    }
}

__kernel void calc_y_gradient(__global float* restrict calc_y_gradient_input_image) {
    float calc_y_gradient_input_image_reuse[5* 1024];
    int32_t g_w[5];
    g_w[0] = 1;
    g_w[1] = -8;
    g_w[2] = 0;
    g_w[3] = 8;
    g_w[4] = 1;
    for (int32_t y_reuse = 0; y_reuse < 436; ++y_reuse) {
      for (int32_t x = 0; x < 1024; ++x) {
        for (int32_t calc_y_gradient_input_image_1 = 0; calc_y_gradient_input_image_1 < 4; ++calc_y_gradient_input_image_1) {
          calc_y_gradient_input_image_reuse[(x + (calc_y_gradient_input_image_1 * 1024))] = calc_y_gradient_input_image_reuse[((x + (calc_y_gradient_input_image_1 * 1024)) + 1024)];
        }
        calc_y_gradient_input_image_reuse[(x + 4096)] = calc_y_gradient_input_image[(x + (y_reuse * 1024))];
        if (4 <= y_reuse) {
          float reducer1;
          reducer1 = 0.000000e+00f;
          for (int32_t rdy = 0; rdy < 5; ++rdy) {
            reducer1 = ((calc_y_gradient_input_image_reuse[(x + (rdy * 1024))] * ((float)g_w[rdy])) + reducer1);
          }
          write_channel_intel(c_buf_2, reducer1);
        }
      }
    }
}

__kernel void calc_x_gradient(__global float* restrict calc_x_gradient_input_image) {
    float calc_x_gradient_input_image_reuse[1*5];
    int32_t g_w[5];
    g_w[0] = 1;
    g_w[1] = -8;
    g_w[2] = 0;
    g_w[3] = 8;
    g_w[4] = 1;
    for (int32_t y = 0; y < 436; ++y) {
      for (int32_t x_reuse = 0; x_reuse < 1024; ++x_reuse) {
        for (int32_t calc_x_gradient_input_image_0 = 0; calc_x_gradient_input_image_0 < 4; ++calc_x_gradient_input_image_0) {
          calc_x_gradient_input_image_reuse[calc_x_gradient_input_image_0] = calc_x_gradient_input_image_reuse[(calc_x_gradient_input_image_0 + 1)];
        }
        calc_x_gradient_input_image_reuse[4] = calc_x_gradient_input_image[(x_reuse + (y * 1024))];
        if (4 <= x_reuse) {
          float reducer0;
          reducer0 = 0.000000e+00f;
          for (int32_t rdx = 0; rdx < 5; ++rdx) {
            reducer0 = ((calc_x_gradient_input_image_reuse[rdx] * ((float)g_w[rdx])) + reducer0);
          }
          write_channel_intel(c_buf_1, reducer0);
        }
      }
    }
}

