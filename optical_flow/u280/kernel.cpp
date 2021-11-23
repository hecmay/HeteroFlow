#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <math.h>

// typedef ap_fixed<32, 20> dtype;
typedef ap_fixed<32, 22> dtype;

typedef struct {
    dtype x;
    dtype y;
} two;

typedef struct {
    dtype x;
    dtype y;
    dtype z;
} triple;

typedef struct {
    dtype a;
    dtype b;
    dtype c;
    dtype d;
    dtype e;
    dtype f;
} sextuple;

static void flow_calc(sextuple* flow_calc_tensor, two* flow_calc_output) {
    for (ap_int<32> y = 0; y < 436; ++y) {
      for (ap_int<32> x = 0; x < 1024; ++x) {
      #pragma HLS pipeline
        sextuple data = flow_calc_tensor[(x + (y * 1024))];
        if ((((2 <= y) && (y < 434)) && (2 <= x)) && (x < 1022)) {
          dtype t0 = data.a;
          dtype t1 = data.b;
          dtype t2 = data.c;
          dtype t3 = data.d;
          dtype t4 = data.e;
          dtype t5 = data.f;
          ap_fixed<64,48> denom = t1 * t2 - t4 * t4;
          ap_fixed<64,48> num0 = ((ap_fixed<64,48>)(( (t5*t3 - t4*t1))));
          ap_fixed<64,48> num1 = ((ap_fixed<64,48>)(( (t4*t3 - t5*t0))));
          if (denom != 0) {
            ap_fixed<64,48> r0 = num0 / denom;
            ap_fixed<64,48> r1 = num1 / denom;
            two res = {(dtype)r0, (dtype)r1};
            flow_calc_output[((x + (y * 1024)))] = res;
          }
        }
      }
    }
  }

static void tensor_weight_y(sextuple* tensor_weight_y_out_product, sextuple* tensor_weight_y_tensor_y) {
    sextuple tensor_weight_y_out_product_reuse[3][1024];
    #pragma HLS array_partition variable=tensor_weight_y_out_product_reuse complete dim=1

    const dtype t_w[3] = {3.243000e-01f, 3.513000e-01f, 3.243000e-01f};
    for (ap_int<32> y_reuse = 0; y_reuse < 436; ++y_reuse) {
      for (ap_int<32> x = 0; x < 1024; ++x) {
      #pragma HLS pipeline
      #pragma HLS dependence variable=tensor_weight_y_out_product_reuse inter false
        for (ap_int<32> k = 0; k < 2; ++k) {
          tensor_weight_y_out_product_reuse[k][x] = tensor_weight_y_out_product_reuse[(k + 1)][x];
        }
        sextuple data = tensor_weight_y_out_product[(x + (y_reuse * 1024))];
        tensor_weight_y_out_product_reuse[2][x] = data; 
        dtype reducer19 = 0;
        dtype reducer18 = 0;
        dtype reducer17 = 0;
        dtype reducer16 = 0;
        dtype reducer15 = 0;
        dtype reducer14 = 0;
        for (ap_int<32> rdx_y = 0; rdx_y < 3; ++rdx_y) {
          sextuple pack = tensor_weight_y_out_product_reuse[(rdx_y)][x];
          reducer14 = ((dtype)((((dtype)pack.a) * t_w[rdx_y]) + ((dtype)reducer14)));
          reducer15 = ((dtype)((((dtype)pack.b) * t_w[rdx_y]) + ((dtype)reducer15)));
          reducer16 = ((dtype)((((dtype)pack.c) * t_w[rdx_y]) + ((dtype)reducer16)));
          reducer17 = ((dtype)((((dtype)pack.d) * t_w[rdx_y]) + ((dtype)reducer17)));
          reducer18 = ((dtype)((((dtype)pack.e) * t_w[rdx_y]) + ((dtype)reducer18)));
          reducer19 = ((dtype)((((dtype)pack.f) * t_w[rdx_y]) + ((dtype)reducer19)));
        }
        sextuple res = {reducer14, reducer15, reducer16, reducer17, reducer18, reducer19};
        tensor_weight_y_tensor_y[((x + (y_reuse * 1024)))] = res;
      }
    }
  }

static void tensor_weight_x(sextuple* tensor_weight_x_tensor_y, sextuple* tensor_weight_x_tensor) {
    sextuple tensor_weight_x_tensor_y_reuse[1][3];
    #pragma HLS array_partition variable=tensor_weight_x_tensor_y_reuse complete dim=2
    const dtype t_w[3] = {3.243000e-01f, 3.513000e-01f, 3.243000e-01f};

    for (ap_int<32> y = 0; y < 436; ++y) {
      for (ap_int<32> x_reuse = 0; x_reuse < 1024; ++x_reuse) {
        for (ap_int<32> k = 0; k < 2; ++k) {
          tensor_weight_x_tensor_y_reuse[0][k] = tensor_weight_x_tensor_y_reuse[0][(k + 1)];
        }
        sextuple data = tensor_weight_x_tensor_y[(x_reuse + (y * 1024))];
        tensor_weight_x_tensor_y_reuse[0][2] = data; 
        dtype reducer13 = 0;
        dtype reducer12 = 0;
        dtype reducer11 = 0;
        dtype reducer10 = 0;
        dtype reducer9  = 0;
        dtype reducer8  = 0;
        for (ap_int<32> rdx_x = 0; rdx_x < 3; ++rdx_x) {
          sextuple pack = tensor_weight_x_tensor_y_reuse[0][(rdx_x)];
          reducer8  = ((dtype)((((dtype)pack.a) * t_w[rdx_x]) + ((dtype)reducer8)));
          reducer9  = ((dtype)((((dtype)pack.b) * t_w[rdx_x]) + ((dtype)reducer9)));
          reducer10 = ((dtype)((((dtype)pack.c) * t_w[rdx_x]) + ((dtype)reducer10)));
          reducer11 = ((dtype)((((dtype)pack.d) * t_w[rdx_x]) + ((dtype)reducer11)));
          reducer12 = ((dtype)((((dtype)pack.e) * t_w[rdx_x]) + ((dtype)reducer12)));
          reducer13 = ((dtype)((((dtype)pack.f) * t_w[rdx_x]) + ((dtype)reducer13)));
        }
        sextuple res = {reducer8, reducer9, reducer10, reducer11, reducer12, reducer13};
        tensor_weight_x_tensor[((x_reuse + (y * 1024)))] = res;
      }
    }
  }

static void outer_product(triple* outer_product_filt_grad, sextuple* outer_product_out_product) {
    for (ap_int<32> y = 0; y < 436; ++y) {
      for (ap_int<32> x = 0; x < 1024; ++x) {
        triple data = outer_product_filt_grad[(x + (y * 1024))];
        dtype d1 = data.x;
        dtype d2 = data.y;
        dtype d3 = data.z;

        ap_fixed<64, 48> a = d1 * d1; 
        ap_fixed<64, 48> b = d2 * d2; 
        ap_fixed<64, 48> c = d3 * d3; 
        ap_fixed<64, 48> d = d1 * d2; 
        ap_fixed<64, 48> e = d1 * d3; 
        ap_fixed<64, 48> f = d2 * d3; 

        sextuple res = {(dtype)a, (dtype)b, (dtype)c, (dtype)d, (dtype)e, (dtype)f};
        outer_product_out_product[(x + (y * 1024))] = res;
      }
    }
  }

static void grad_weight_x(triple* grad_weight_x_y_filt, triple* grad_weight_x_filt_grad) {
    triple grad_weight_x_y_filt_reuse[1][7];
    #pragma HLS array_partition variable=grad_weight_x_y_filt_reuse complete dim=2

    const dtype g_f[7] = {0.0755, 0.13300, 0.1869, 0.2903,
        0.1869, 0.133, 0.0755};

    for (ap_int<32> y = 0; y < 436; ++y) {
      for (ap_int<32> x_reuse = 0; x_reuse < 1024; ++x_reuse) {
      #pragma HLS pipeline
        for (ap_int<32> k = 0; k < 6; ++k) {
          grad_weight_x_y_filt_reuse[0][k] = grad_weight_x_y_filt_reuse[0][(k + 1)];
        }
        triple data = grad_weight_x_y_filt[(x_reuse + (y * 1024))];
        grad_weight_x_y_filt_reuse[0][6] = data;
        dtype reducer7 = 0;
        dtype reducer6 = 0;
        dtype reducer5 = 0;
        for (ap_int<32> rdx = 0; rdx < 7; ++rdx) {
          triple pack = grad_weight_x_y_filt_reuse[0][(rdx)];
          reducer5 = ((dtype)((((dtype)pack.x) * g_f[rdx]) + ((dtype)reducer5)));
          reducer6 = ((dtype)((((dtype)pack.y) * g_f[rdx]) + ((dtype)reducer6)));
          reducer7 = ((dtype)((((dtype)pack.z) * g_f[rdx]) + ((dtype)reducer7)));
        }
        triple res = {reducer5, reducer6, reducer7};
        grad_weight_x_filt_grad[((x_reuse + (y * 1024)))] = res;
      }
    }
  }


static void grad_weight_y(dtype* grad_weight_y_grad_x, dtype* grad_weight_y_grad_y, dtype* grad_weight_y_grad_z, triple* grad_weight_y_y_filt) {
    triple grad_weight_y_grad_reuse[7][1024];
    #pragma HLS array_partition variable=grad_weight_y_grad_reuse complete dim=1

    const dtype g_f[7] = {7.550000e-02f, 1.330000e-01f, 1.869000e-01f,
        2.903000e-01f,
        1.869000e-01f, 1.330000e-01f, 7.550000e-02f};

    for (ap_int<32> y_reuse = 0; y_reuse < 436; ++y_reuse) {
      for (ap_int<32> x = 0; x < 1024; ++x) {
      #pragma HLS pipeline
      #pragma HLS dependence variable=grad_weight_y_grad_reuse inter false
        for (ap_int<32> k = 0; k < 6; ++k) {
          grad_weight_y_grad_reuse[k][x] = grad_weight_y_grad_reuse[(k + 1)][x];
        }
        triple data = {
            grad_weight_y_grad_x[(x + (y_reuse * 1024))],
            grad_weight_y_grad_y[(x + (y_reuse * 1024))],
            grad_weight_y_grad_z[(x + (y_reuse * 1024))]
        };
        grad_weight_y_grad_reuse[6][x] = data; 
        dtype reducer4 = 0;
        dtype reducer3 = 0;
        dtype reducer2 = 0;
        for (ap_int<32> rdx = 0; rdx < 7; ++rdx) {
          triple pack = grad_weight_y_grad_reuse[(rdx)][x];
          reducer2 = ((dtype)((((dtype)pack.x) * g_f[rdx]) + ((dtype)reducer2)));
          reducer3 = ((dtype)((((dtype)pack.y) * g_f[rdx]) + ((dtype)reducer3)));
          reducer4 = ((dtype)((((dtype)pack.z) * g_f[rdx]) + ((dtype)reducer4)));
        }
        triple res = {reducer2, reducer3, reducer4};
        grad_weight_y_y_filt[((x + (y_reuse * 1024)))] = res;
      }
    }
  }


static void calc_z_gradient(ap_uint<8>* calc_z_gradient_img0, ap_uint<8>* calc_z_gradient_img1, ap_uint<8>* calc_z_gradient_img2_0, ap_uint<8>* calc_z_gradient_img3, ap_uint<8>* calc_z_gradient_img4, dtype* calc_z_gradient_grad_z) {
    const ap_int<32> g_w[5] = {1, -8, 0, 8, 1};
    for (ap_int<32> y = 0; y < 436; ++y) {
      for (ap_int<32> x = 0; x < 1024; ++x) {
      #pragma HLS pipeline
        calc_z_gradient_grad_z[(x + (y * 1024))] = ((dtype)(((dtype)(((ap_fixed<68, 56>)(((ap_fixed<67, 55>)(((ap_fixed<66, 54>)(((ap_fixed<65, 53>)(((ap_fixed<64, 52>)calc_z_gradient_img0[(x + (y * 1024))]) * ((ap_int<64>)g_w[0]))) + ((ap_fixed<65, 53>)(((ap_fixed<64, 52>)calc_z_gradient_img1[(x + (y * 1024))]) * ((ap_int<64>)g_w[1]))))) + ((ap_fixed<66, 54>)(((ap_fixed<64, 52>)calc_z_gradient_img2_0[(x + (y * 1024))]) * ((ap_int<64>)g_w[2]))))) + ((ap_fixed<67, 55>)(((ap_fixed<64, 52>)calc_z_gradient_img3[(x + (y * 1024))]) * ((ap_int<64>)g_w[3]))))) + ((ap_fixed<68, 56>)(((ap_fixed<64, 52>)calc_z_gradient_img4[(x + (y * 1024))]) * ((ap_int<64>)g_w[4]))))) / 12));
      }
    }
  }

static void calc_xy_gradient(ap_uint<8>* calc_xy_gradient_input_image, dtype* calc_xy_gradient_grad_x, dtype* calc_xy_gradient_grad_y) {
    ap_uint<8> calc_xy_gradient_input_image_1_reuse[5][1024];
    #pragma HLS array_partition variable=calc_xy_gradient_input_image_1_reuse complete dim=1
    ap_uint<8> calc_xy_gradient_input_image_0_reuse[1][5];
    #pragma HLS array_partition variable=calc_xy_gradient_input_image_0_reuse complete dim=2

    const ap_int<32> g_w[5] = {1, -8, 0, 8, 1};

    for (ap_int<32> y_reuse = 0; y_reuse < 436; ++y_reuse) {
      for (ap_int<32> x = 0; x < 1024; ++x) {
      #pragma HLS pipeline
      #pragma HLS dependence variable=calc_y_gradient_input_image_1_reuse inter false
        for (ap_int<32> k = 0; k < 4; ++k) {
          calc_xy_gradient_input_image_0_reuse[0][k] = calc_xy_gradient_input_image_0_reuse[0][(k + 1)];
          calc_xy_gradient_input_image_1_reuse[k][x] = calc_xy_gradient_input_image_1_reuse[(k + 1)][x];
        }
        ap_uint<8> data = calc_xy_gradient_input_image[(x + (y_reuse * 1024))];
        calc_xy_gradient_input_image_1_reuse[4][x] = data;
        calc_xy_gradient_input_image_0_reuse[0][4] = data;
        dtype reducer1;
        dtype reducer0;
        reducer0 = ((dtype)0);
        reducer1 = ((dtype)0);
        for (ap_int<32> rdy = 0; rdy < 5; ++rdy) {
          reducer0 = ((dtype)(((ap_fixed<65, 53>)(((ap_fixed<64, 52>)calc_xy_gradient_input_image_0_reuse[0][(rdy)]) * ((ap_int<64>)g_w[rdy]))) + ((ap_fixed<65, 53>)reducer0)));
          reducer1 = ((dtype)(((ap_fixed<65, 53>)(((ap_fixed<64, 52>)calc_xy_gradient_input_image_1_reuse[(rdy)][x]) * ((ap_int<64>)g_w[rdy]))) + ((ap_fixed<65, 53>)reducer1)));
        }
        calc_xy_gradient_grad_x[((x + (y_reuse * 1024)))] = reducer0;
        calc_xy_gradient_grad_y[((x + (y_reuse * 1024)))] = reducer1;
      }
    }
  }

extern "C" {
    void test(ap_uint<64>* input_images, two* output_image) {
    #pragma HLS INTERFACE m_axi port=input_images offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=output_image offset=slave bundle=gmem1

    #pragma HLS INTERFACE s_axilite port=input_images bundle=control
    #pragma HLS INTERFACE s_axilite port=output_image bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    #pragma HLS aggregate variable=output_image

    #pragma HLS dataflow

      ap_uint<8> frame1[446464];
      #pragma HLS stream variable=frame1 depth=1024
      ap_uint<8> frame2[446464];
      #pragma HLS stream variable=frame2 depth=1024
      ap_uint<8> frame3_a[446464];
      #pragma HLS stream variable=frame3_a depth=1024
      ap_uint<8> frame3_b[446464];
      #pragma HLS stream variable=frame3_b depth=1024
      ap_uint<8> frame4[446464];
      #pragma HLS stream variable=frame4 depth=1024
      ap_uint<8> frame5[446464];
      #pragma HLS stream variable=frame5 depth=1024

      for (int r=0; r < 436; r++) {
         for (int c=0; c < 1024; c++) {
         #pragma HLS pipeline II=1
             ap_int<64> buf = input_images[r * 1024 + c];
             frame1[r * 1024 + c]   = ((buf(7,   0)) >> 8);
             frame2[r * 1024 + c]   = ((buf(15,  8)) >> 8);
             frame3_a[r * 1024 + c] = ((buf(23, 16)) >> 8);
             frame3_b[r * 1024 + c] = ((buf(23, 16)) >> 8);
             frame4[r * 1024 + c]   = ((buf(31, 24)) >> 8);
             frame5[r * 1024 + c]   = ((buf(39, 32)) >> 8);
         }
      }

      dtype grad_x[446464];
      #pragma HLS stream variable=grad_x depth=1024
      dtype grad_y[446464];
      #pragma HLS stream variable=grad_y depth=1024
      calc_xy_gradient(frame3_a, grad_x, grad_y);

      dtype grad_z[446464];
      #pragma HLS stream variable=grad_z depth=1024*4
      calc_z_gradient(frame1, frame2, frame3_b, frame4, frame5, grad_z);

      triple y_filt[446464];
      #pragma HLS stream variable=y_filt depth=1024
      #pragma HLS aggregate variable=y_filt
      grad_weight_y(grad_x, grad_y, grad_z, y_filt);

      triple filt_grad[446464];
      #pragma HLS stream variable=filt_grad depth=1024
      #pragma HLS aggregate variable=filt_grad
      grad_weight_x(y_filt, filt_grad);
      
      sextuple out_product[446464];
      #pragma HLS stream variable=out_product depth=1024
      #pragma HLS aggregate variable=out_product
      outer_product(filt_grad, out_product);

      sextuple tensor_y[446464];
      #pragma HLS stream variable=tensor_y depth=1024
      #pragma HLS aggregate variable=tensor_y
      tensor_weight_y(out_product, tensor_y);

      sextuple tensor[446464];
      #pragma HLS stream variable=tensor depth=1024
      #pragma HLS aggregate variable=tensor
      tensor_weight_x(tensor_y, tensor);

      flow_calc(tensor, output_image);
    }
}
