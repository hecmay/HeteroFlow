CINCLUDES += -I/opt/xilinx/Xilinx_Vivado_vitis_2019.2/Vivado/2019.2/include/
CXXFLAGS += -std=c++11 -Wall -g
CXXFLAGS += -Wno-unknown-pragmas
CXXFLAGS += -Wno-unused-function
CXXFLAGS += -Wno-unused-variable
CXXFLAGS += -Wno-misleading-indentation
CXXFLAGS += -Wno-int-in-bool-context
CXX = g++
LDFLAGS = -lpthread

HLS = vivado_hls -f

PROJ_ROOT = ./
IMG_DIR = $(PROJ_ROOT)/data/example_images
CSIM_DIR = $(PROJ_ROOT)/ultranet_csim
COSIM_DIR = $(PROJ_ROOT)/ultranet_cosim

$(CSIM_DIR)/ultranet_csim_exe:
	mkdir -p $(CSIM_DIR)
	$(CXX) $(CXXFLAGS) $(CINCLUDES) csim_tb.cpp ultranet.cpp -o ultranet_csim_exe $(LDFLAGS)
	mv ultranet_csim_exe $(CSIM_DIR)

csim: $(CSIM_DIR)/ultranet_csim_exe
	python3 csim.py $(IMG_DIR)/000001.jpg $(CSIM_DIR)

cosim:
	$(HLS) cosim.tcl presim

build:
	$(HLS) ultranet_hls.tcl

clean:
	rm -rf $(CSIM_DIR) $(COSIM_DIR)

cleanall: clean
	rm -rf ultranet
