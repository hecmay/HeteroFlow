############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
############################################################c
open_project ultranet
set_top ultranet
add_files systolic_array.h
add_files ultranet.cpp
add_files stream_tools.h
add_files sliding_window_unit.h
add_files pool2d.h
add_files param.h
add_files matrix_vector_unit.h
add_files function.h
add_files conv2d.h
add_files config.h
#add_files -tb csim_tb.cpp
add_files bn_qrelu2d.h
open_solution "solution1"
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 5 -name default
#source "./ultranet/solution1/directives.tcl"
# csim_design
csynth_design
# cosim_design
export_design -rtl verilog -format ip_catalog

exit
