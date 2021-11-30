#pragma once

#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;
#include "stream_tools.h"



template <	unsigned K,
			unsigned S,
			unsigned Din_H,
			unsigned Din_W,
			unsigned Cin,
			unsigned Ibit>
void SWU(
	stream<ap_uint<Cin*Ibit> >& in,
	stream<ap_uint<Cin*Ibit> >& out,
	const unsigned reps = 1)
{

	const unsigned steps = (Din_W-K)/S+1;
	const unsigned line_buffer_size = K*Din_W;
#ifdef SWU_DEBUG
	cout << "steps: " << steps << endl;
	cout << "line_buffer_size: " << line_buffer_size << endl;
#endif

	ap_uint<Cin*Ibit> line_buffer[line_buffer_size];
#pragma HLS RESOURCE variable line_buffer core=RAM_2P

	ap_uint<Cin*Ibit> temp_in;

	ap_uint<1> initial_fill = 0;
	unsigned stride = 0;
	unsigned pointer = 0;
	unsigned h = 0;

	for (unsigned rep = 0; rep < reps*Din_H; rep++) {
		#pragma HLS loop_tripcount min=40 max=320
		if (h == Din_H) {
			initial_fill = 0;
			stride = 0;
			pointer = 0;
			h = 0;
		}
		h += 1;

#ifdef SWU_DEBUG
		cout << "wpointer: " << pointer << endl;
#endif

		for (unsigned w = 0; w < Din_W; w++) {
			#pragma HLS loop_tripcount min=20 max=640
#pragma HLS PIPELINE II=1
			temp_in = in.read();

			unsigned line_buffer_pointer = pointer + w;
			if (line_buffer_pointer >= line_buffer_size) {
				line_buffer_pointer = line_buffer_pointer - line_buffer_size;
			}
#ifdef SWU_DEBUG
			cout << "line_buffer_pointer: " << line_buffer_pointer << endl;
#endif
			line_buffer[line_buffer_pointer] = temp_in;
		}

		stride += 1;
		pointer += Din_W;
		if (pointer >= line_buffer_size) {
			pointer = pointer - line_buffer_size;
			initial_fill = 1;
#ifdef SWU_DEBUG
			cout << "initial_fill set to 1!" << endl;
#endif
		}

#ifdef SWU_DEBUG
		cout << "stride: " << stride << endl;
		cout << "rpointer: " << pointer << endl;
		cout << "line_buffer for out: ";
		for (unsigned j = 0; j < line_buffer_size; j++) {
			#pragma HLS loop_tripcount min=9 max=25
			cout << line_buffer[j] << " ";
		}
		cout << endl;
#endif
		if (initial_fill == 1 && stride >= S) {
			stride = 0;

			unsigned s = 0;
			unsigned x = 0;
			unsigned y = 0;

			for (unsigned i = 0; i < steps*(K*K); i++ ) {
				#pragma HLS loop_tripcount min=9 max=25
#pragma HLS PIPELINE II=1
				unsigned read_address = (pointer+s*S) + y*Din_W + x;

				if (read_address >= line_buffer_size)
					read_address = read_address - line_buffer_size;
#ifdef SWU_DEBUG
				cout << "read_address: " << read_address << endl;
#endif
				ap_uint<Cin*Ibit> temp_out = line_buffer[read_address];
				out.write(temp_out);

				if (x == K-1) {
					x = 0;
					if (y == K-1) {
						y = 0;
						if (s == steps-1)
							s = 0;
						else
							s++;
					}
					else
						y++;
				}
				else
					x++;
			}
		}
	}
}

template <	unsigned K,
			unsigned S,
			unsigned IN_ROW,
			unsigned IN_COL,
			unsigned IN_CH,
			unsigned IN_BIT>
void sliding_window_unit(
	stream<ap_uint<IN_CH*IN_BIT> >& in,
	stream<ap_uint<IN_CH*IN_BIT> >& out,
	const unsigned reps = 1)
{
	// 行方向上需要移动多少次 向下移动次数
    // total number of downwards hops (on the row dimension)
	const unsigned ROW_STEPS = (IN_ROW-K) / S + 1;
	// 想右移动次数
    // total number of rightwards hops (on the column dimension)
	const unsigned COL_STEPS = (IN_COL-K) / S + 1;

	// TODO buf应该还可以优化
	// 当图像尺寸不一致时 选用 row优先 or col优先应该对 这里的buff消耗有影响
	// 例如当 K = 3时 实际上不需要 完整的 3行来缓存 而是只需要 2 × IN_COL + 3就可以解除依赖
	// 构建一个循环列队
    // TODO: possibly we can further optimize the buffer
    // when the image is not a square, choosing to move along the row dimension first or the column dimension first should effect the buffer size
    // e.g., when K = 3, we do not need to buffer 3 entire rows of the image, but buffering only 2 x IN_COL + 3 pixels is enough to resolve dependency
    // construct a cyclic queue
	const unsigned BUF_SIZE = (K - 1) * IN_COL + K;
	ap_uint<IN_CH*IN_BIT> line_buffer[BUF_SIZE];
#pragma HLS RESOURCE variable line_buffer core=RAM_2P
	unsigned buf_len = 0;
	unsigned buf_pointer = 0;
	ap_uint<IN_CH*IN_BIT> temp_in;

	// 滑动计数
    // hop count
	unsigned right_slid = 0;
	unsigned down_slid = 0;
	// 一共循环的次数
    // total loop trip count
	for(unsigned rep=0; rep < IN_ROW*IN_COL*reps; rep ++) {
		#pragma HLS loop_tripcount min=20*40 max=360*640
		// 写数据到 buf
		// buf 不满的时候一直写数据
        // write data into the buffer,
        // until it's full
		if(buf_len < BUF_SIZE) {
			// TODO
			temp_in = in.read();
			line_buffer[buf_pointer++] = temp_in;
			if(buf_pointer == BUF_SIZE) {
				buf_pointer = 0;
			}
			buf_len ++;
		}

		// 缓冲区满 可以输出数据
        // the buffer is full, ready to output
		if(buf_len == BUF_SIZE) {
			// 输出窗口数据
			// 缓冲区寻址 pointer 指向的是下一个位置
			// 如果规定每来一个元素都是放在队头，当i=0时 pointer实际指向的元素是最后一个元素
			// 而这个元素正是这里要最先输出的
            // output windowed data
            // the buffer address "buf_pointer" points to the next available position in the queue
            // so if we assume that every new element will be placed at the head of the queue, when i = 0, the pointer actually points to the last element in the queue
            // and that element is the first one to output
			for(unsigned i=0; i < K; i ++) {
				#pragma HLS loop_tripcount min=3 max=5
				for(unsigned j=0; j < K; j ++) {
					#pragma HLS loop_tripcount min=3 max=5
					// 寻址
                    // calculate the address
					unsigned temp_pointer = (buf_pointer + (i * IN_COL) + j);
					// 这里temp_pointer 不可能大于 2 × BUF_SIZE
                    // temp_pointer cannot be greater than 2 x BUF_SIZE
					if(temp_pointer > BUF_SIZE) {
						temp_pointer -= BUF_SIZE;
					}

					ap_uint<IN_CH*IN_BIT> temp_out = line_buffer[temp_pointer];
					out.write(temp_out);
				}
			}
			// 输出后窗口向右滑动
            // hop rightwards when output finished

			// 滑到头了
            // reached theright end
			if(++ right_slid == COL_STEPS) {
				right_slid = 0;
				// 右滑到头 下滑
                // hop downwards
				if(++ down_slid == ROW_STEPS) {
					down_slid = 0;
					// 一帧数据完
                    // finish processing one input frame (hit the bottom end)
					buf_len = 0;
				} else {
					// 下滑没有到头
                    // not reach the bottom end yet
					buf_len = buf_len - (S-1) * IN_COL - K;
				}
			} else {
				// 右滑没到头
				// S 个数据 出缓冲
                // not reach the end yet
                // move S data packets out of the buffer
				buf_len -= S;
			}
		}
	}
}

