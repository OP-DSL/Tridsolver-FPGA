### High-Level Batched TriDiagonal Systems solver Library for Xilinx FPGAs
This Tridiagonal Library can be used in Multi Dimensional Implicit numerical applications where solving multiple Tridiagonal sytems is common. Thomas algorithm is used to solve interleaved tridiagonal systems inorder to get high performance on Xilinx FPGAs. We further developed Tiled Tridiagonal solver for larger systems. This library is developed targetting current generation Xilinx Accelerator cards with HBM memory to scale compute units. We have tested the library on Xilinx Alveo U280 and Xilinx Alveo U50 devices.  

#### Representative applications
We tested the 2D and 3D Heat diffusion application using FP32 and FP64 arithmetics. All of our application implementation supports the Batched computation of the meshes. Another varient of applications for 2D Heat diffusion application on larger meshes also developed using Tiled thomas solvers. Following application can be found in /FPGA directory  

* ADI2D_F32       - 2D Heat diffusion application using FP32
* ADI2D_F64       - 2D Heat diffusion application using FP64
* ADI3D_F32       - 3D Heat diffusion application using FP32
* ADI3D_F64       - 2D Heat diffusion application using FP64
* ADI2D_TH_TH_F32 - 2D Heat diffusion application on larger meshes using FP32, using Thomas-Thomas solver
* ADI2D_THPCR_F32 - 2D Heat diffusion application on larger meshes using FP32, using Thomas-Thomas solver



#### Application Implementations  
Vitis flow is used for implmenting applications targetting XIlinx Alveo U280 and U50 devices. Makefile will be added in future to support commandline based implementation. 
Kernel files are named as *_kernel.cpp.

each application folder contains configuration files named as *\*.cfg* and necessary placement and memory port constraints are provided there. 
you can set constriants in vitis GUI flow as *--config \*.cfg* in the GUI command box of the binray container and kernels.
