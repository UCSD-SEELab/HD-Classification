VPP := $(XILINX_VITIS)/bin/v++
EMCONFIGUTIL := $(XILINX_VITIS)/bin/emconfigutil
MODE := hw
#MODE := sw_emu
PLATFORM := xilinx_u280_xdma_201920_1

# sources
KERNEL_SRC := src/hd.cpp
KERNEL_HEADER := src/hd.h
HOST_SRC := src/host.cpp

# targets
HOST_EXE := host.exe

XOS := hd.$(MODE).xo
XCLBIN := hd.$(MODE).xclbin
EMCONFIG_FILE := emconfig.json

# Linker options to map kernel ports to DDR banks
#VPP_LINK_OPTS := --config connectivity.cfg
VPP_LINK_OPTS := --kernel_frequency 100

VPP_COMMON_OPTS := -s -t $(MODE) --platform $(PLATFORM) --hls.clock 100000000:hd
CFLAGS := -g -std=c++11 -I$(XILINX_XRT)/include
LFLAGS := -L$(XILINX_XRT)/lib -lxilinxopencl -lpthread -lrt
NUMDEVICES := 1

# run time args
EXE_OPT := hd.$(MODE).xclbin

# primary build targets
.PHONY: xclbin app all

xclbin:  $(XCLBIN)
app: $(HOST_EXE)

all: xclbin app

clean:
	-$(RM) $(HOST_EXE)

# kernel rules
$(XOS): $(KERNEL_SRC)
$(XOS): $(KERNEL_SRC) $(KERNEL_HEADER)
	$(RM) $@
	$(VPP) $(VPP_COMMON_OPTS) -c -k hd -o $@ $+

$(XCLBIN): $(XOS)
	$(VPP) $(VPP_COMMON_OPTS) -l -o $@ $+ $(VPP_LINK_OPTS)

# host rules
$(HOST_EXE): $(HOST_SRC)
	g++ $(CFLAGS) -o $@ $+ $(LFLAGS)
	@echo 'Compiled Host Executable: $(HOST_EXE)'

$(EMCONFIG_FILE):
	$(EMCONFIGUTIL) --nd $(NUMDEVICES) --od . --platform $(PLATFORM)

check: $(XCLBIN) $(HOST_EXE) $(EMCONFIG_FILE)
	XCL_EMULATION_MODE=${MODE} ./$(HOST_EXE) $(EXE_OPT)
