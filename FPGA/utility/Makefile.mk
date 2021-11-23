#
# Copyright 2019-2021 Xilinx, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# makefile-generator v1.0.3


PLATFORM=/opt/xilinx/platforms/xilinx_u280_xdma_201920_3/xilinx_u280_xdma_201920_3.xpfm
TARGET=sw_emu

CC=g++
CC_CFLAGS=-Wall -O0 -g -std=c++11
CC_LFLAGS=-L${XILINX_XRT}/lib/ -lOpenCL -lxilinxopencl -lpthread -lrt -lstdc++ -fopenmp -DVITIS_PLATFORM=xilinx_u280_xdma_201920_3
CC_INCDIR=-I${XILINX_XRT}/include/ -I${XILINX_VIVADO}/include/ -I../utility/

KCC=v++
KCC_CFLAGS=--platform $(PLATFORM) --config $(CONFIG)
KCC_LFLAGS=--platform $(PLATFORM) --config $(CONFIG)
KCC_INCDIR=-I../lib/


BUILD_DIR=./build/
TMP_DIR=./tmp

APP=$(BUILD_DIR)adi_app.exe


############################## Help Section ##############################
.PHONY: help

help::
	@echo "Makefile Usage:"
	@echo "  make all TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	@echo "      Command to generate the design for specified Target and Shell."
	@echo ""
	@echo "  make clean "
	@echo "      Command to remove the generated files."
	@echo ""
	@echo "  make run TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	@echo "      Command to run application in emulation."
	@echo ""
	@echo "  make build TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	@echo "      Command to build xclbin application."
	@echo ""



build: $(APP) $(BUILD_DIR)$(K_SRC)_$(TARGET).xclbin


$(BUILD_DIR)%.exe: $(HOST_SRC)
	mkdir -p $(BUILD_DIR)
	$(CC) $(CC_INCDIR) $(CC_CFLAGS)  $(HOST_SRC) -o $@ $(CC_LFLAGS)

$(BUILD_DIR)%.xo: %.cpp
	mkdir -p $(TMP_DIR)
	$(KCC) -t $(TARGET) $(KCC_CFLAGS) --temp_dir $(TMP_DIR) -c -k TDMA_batch $(KCC_INCDIR) -o $@ $<

$(BUILD_DIR)%_$(TARGET).xclbin: $(BUILD_DIR)%.xo
	mkdir -p $(BUILD_DIR)
	$(KCC) -t $(TARGET) $(KCC_LFLAGS) --temp_dir $(BUILD_DIR) -l -g -o $@ $<




run: build
ifeq ($(TARGET), sw_emu)
	XCL_EMULATION_MODE=$(TARGET) $(APP) $(BUILD_DIR)$(K_SRC)_$(TARGET).xclbin $(APP_ARGS)
else
	$(APP)  ./$(BUILD_DIR)$(K_SRC)_$(TARGET).xclbin $(APP_ARGS)
endif


clean: 
	rm -rf $(BUILD_DIR) $(TMP_DIR)


