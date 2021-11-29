### Batched Tridiagonal Systems Solver Library for Xilinx FPGAs
The Tridsolver-FPGA Library provides high-throughput implementations of multiple multi-dimensional tridiagonal system solvers on FPGAs. The libray is based on the inexpensive Thomas algorithm with batching of multiple systems for solving smaller and medium sized systems and hybrid Thomas_PCR and Thomas_Thomas algorithms to solve larger systems. The HLS Techniques used to implement the Libray and data path for 3D ADI applications can be found [here](https://github.com/Kamalavasan/Tridsolver-FPGA/blob/main/Docs/HLS_Techniques.md). The library currentry only supports Xilinx FPGA devices and have been tested on Xilinx Alveo U280 and Alveo U50 cards. Ongoing and future work will add support for Intel FPGAs. The library and performance results are currenty under review for publication. 

#### Representative applications
The library has been used to implement the 2D and 3D Heat diffusion application using FP32 and FP64 arithmetic. The implementation supports the batched computation of systems. The `/FPGA` directory consists the following varients of these applications. 

<table>
<!--   <caption>2D ADI Heat Diffusion Application Performance, 120 iter</caption> -->
  <tr>
    <td> ADI2D_F32 </td>
    <td>2D ADI application using FP32 </td>
  </tr>
  <tr>
    <td> ADI2D_F32 </td>
    <td>2D ADI application using FP64 </td>
  </tr>
  <tr>
    <td> ADI3D_F32 </td>
    <td>3D ADI application using FP32 </td>
  </tr>
  <tr>
    <td> ADI3D_F32 </td>
    <td>3D ADI application using FP64 </td>
  </tr>
  <tr>
    <td> ADI2D_TH_TH_F32 </td>
    <td>2D ADI application with Tiled Thomas-Thomas solver using FP32 </td>
  </tr>
  <tr>
    <td> ADI2D_THPCR_F32 </td>
    <td>2D ADI application with Tiled Thomas-PCR solver using FP32 </td>
  </tr>
 </table>


#### Application Implementations  
Makefile based FPGA application implementation is supported. Optionally user can implement Application using Vitis GUI. In that case, user need to point the config file and set number of kernels. Here we note that separate config files are provided for U50 and U280 devices. 

Following are the steps for Makefile based flow, 

`cd <application directory>`

`set the target config file(_u50.cfg or u280.cfg) in the Makefile`

` make build TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform>`

` make run TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform>`

#### Performance comparison of Xilinx Accelaration Cards with Nvidia V100 GPU
The performance of Tridsolver-FPGA library on Xilinx FPGAs has been compared to performance of the same applications on Nvida V100 GPUs (using the [Tridsolver GPU library by László et al.](https://github.com/OP-DSL/tridsolver) and NVIDIA's cuSPARSE). The following results are for the 2D and 3D Heat Diffusion Application implemented with the ADI technique and a Stochastic Local Volatility (SLV) model application, implemented with a Hundsdorfer-Verwer (HV) method for time integration. 

### Xilinx Alveo U50 Vs Nvidia V100

<table>
  <caption>2D ADI Heat Diffusion Application Performance, 120 iter</caption>
  <tr>
    <td><img src="https://github.com/Kamalavasan/Tridsolver-FPGA/blob/main/Results/Graph/ADI-2D-SP_log_U50.svg" width=500 </td>
    <td><img src="https://github.com/Kamalavasan/Tridsolver-FPGA/blob/main/Results/Graph/ADI-2D-DP_log_U50.svg" width=500 </td>
  </tr>
   <tr>
     <td>FP32, v= 8,  f<sub>CU</sub>=3,   N<sub>CU</sub>=2 </td>
     <td>FP64, v= 8,  f<sub>CU</sub>=3,   N<sub>CU</sub>=2 </td>
  </tr>
 </table>
 
 <table>
  <caption>3D ADI Heat Diffusion Application Performance, 100 iter</caption>
  <tr>
    <td><img src="https://github.com/Kamalavasan/Tridsolver-FPGA/blob/main/Results/Graph/ADI-3D-SP_log_U50.svg" width=500 </td>
    <td><img src="https://github.com/Kamalavasan/Tridsolver-FPGA/blob/main/Results/Graph/ADI-3D-DP_log_U50.svg" width=500 </td>
  </tr>
   <tr>
    <td>FP32, v= 8, N<sub>CU</sub>=4 </td>
    <td>FP64, v= 8, N<sub>CU</sub>=2 </td>
  </tr>
 </table>
 
 
 <table>
  <caption>2D ADI Heat Diffusion Application on Larger Meshes, 100 iter</caption>
  <tr>
    <td><img src="https://github.com/Kamalavasan/Tridsolver-FPGA/blob/main/Results/Graph/ADI-2D-SP_THTH_log_U50.svg" width=500 </td>
    <td><img src="https://github.com/Kamalavasan/Tridsolver-FPGA/blob/main/Results/Graph/ADI-2D-SP-THPCR-U50.svg" width=500 </td>
  </tr>
   <tr>
    <td>FP32, Thomas-Thomas solver, N<sub>CU</sub>=4 </td>
    <td>FP32, Thomas-PCR solver, N<sub>CU</sub>=4  </td>
  </tr>
 </table>
 
 
  <table>
  <caption>SLV Application performance</caption>
  <tr>
    <td><img src="https://github.com/Kamalavasan/Tridsolver-FPGA/blob/main/Results/Graph/SLV-40x20_U50.svg" width=500 </td>
    <td><img src="https://github.com/Kamalavasan/Tridsolver-FPGA/blob/main/Results/Graph/SLV-100x50_U50.svg" width=500 </td>
  </tr>
   <tr>
    <td>40x20 Mesh, v = 1, N<sub>CU</sub>=2, FP64 </td>
    <td>100x50 Mesh, v = 1, N<sub>CU</sub>=2, FP64 </td>
  </tr>
 </table>




## Xilinx Alveo U280 Vs Nvidia V100
<table>
  <caption>2D ADI Heat Diffusion Application Performance, 120 iter</caption>
  <tr>
    <td><img src="https://github.com/Kamalavasan/Tridsolver-FPGA/blob/main/Results/Graph/ADI-2D-SP_log.svg" width=500 </td>
    <td><img src="https://github.com/Kamalavasan/Tridsolver-FPGA/blob/main/Results/Graph/ADI-2D-DP_log.svg" width=500 </td>
  </tr>
   <tr>
     <td>FP32, v= 8,  f<sub>CU</sub>=3,   N<sub>CU</sub>=3 </td>
     <td>FP64, v= 8,  f<sub>CU</sub>=3,   N<sub>CU</sub>=3 </td>
  </tr>
 </table>
 
 <table>
  <caption>3D ADI Heat Diffusion Application Performance, 100 iter</caption>
  <tr>
    <td><img src="https://github.com/Kamalavasan/Tridsolver-FPGA/blob/main/Results/Graph/ADI-3D-SP_log.svg" width=500 </td>
    <td><img src="https://github.com/Kamalavasan/Tridsolver-FPGA/blob/main/Results/Graph/ADI-3D-DP_log.svg" width=500 </td>
  </tr>
   <tr>
    <td>FP32, v= 8, N<sub>CU</sub>=6 </td>
    <td>FP64, v= 8, N<sub>CU</sub>=3 </td>
  </tr>
 </table>
 
 
 <table>
  <caption>2D ADI Heat Diffusion Application on Larger Meshes, 100 iter</caption>
  <tr>
    <td><img src="https://github.com/Kamalavasan/Tridsolver-FPGA/blob/main/Results/Graph/ADI-2D-SP_THTH_log.svg" width=500 </td>
    <td><img src="https://github.com/Kamalavasan/Tridsolver-FPGA/blob/main/Results/Graph/ADI-2D-SP_THPCR_log.svg" width=500 </td>
  </tr>
   <tr>
    <td>FP32, Thomas-Thomas solver,  N<sub>CU</sub>=4 </td>
    <td>FP32, Thomas-PCR solver,  N<sub>CU</sub>=4  </td>
  </tr>
 </table>
 
 
  <table>
  <caption>SLV Application performance</caption>
  <tr>
    <td><img src="https://github.com/Kamalavasan/Tridsolver-FPGA/blob/main/Results/Graph/SLV-40x20.svg" width=500 </td>
    <td><img src="https://github.com/Kamalavasan/Tridsolver-FPGA/blob/main/Results/Graph/SLV-100x50.svg" width=500 </td>
  </tr>
   <tr>
    <td>40x20 Mesh, v = 1, N<sub>CU</sub>=3, FP64 </td>
    <td>100x50 Mesh, v = 1, N<sub>CU</sub>=3, FP64 </td>
  </tr>
 </table>




