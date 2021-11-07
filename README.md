### High-Level Batched TriDiagonal Systems solver Library for Xilinx FPGAs
This Tridiagonal Library can be used to get high performance on FPGA when there is a need to solve multiple TriDiagonal systems. Libray utilise the inexpensive Thomas solver with batching to eliminate the carried loop dependency in thomas solver. One limitation of the batched thomas solver implementation on FPGAs is system size due to limited on chip memory. Thomas_PCR and Thomas_thomas solver Library can be used to solve batch of larger systems. Currrently this library supports only current generation Xilinx Accelaration cards. This libray have been teste on Xilinx Alveo U280 and Xilinx Alveo U50 devices with multiple CUs. Results shows that libray scales well without causing any significant drop in the operating frequency.  

#### Representative applications
We tested the 2D and 3D Heat diffusion application using FP32 and FP64 arithmetics. All of our application implementation supports the Batched computation of the meshes. Legacy thomas solver is used to implement ADI application for smaller meshes. Thomas-Thomas and Thomas-PCR based Tiled Thomas solvers are used in ADI applications for larger meshes. Following application can be found in /FPGA directory. Additionally we tested more complex SLV application using Hundsdorfer-Verwer(HV) method in FP64 arithmetic. Since this application we got from NAG UK, we can't make this opensource. List of Varients we tested as follows.   


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
  <tr>
    <td> SLV </td>
    <td> SLV application using Hundsdorfer-Verwer(HV) in FP64   </td>
  </tr>
 </table>



#### Application Implementations  
Vitis flow is used for implmenting applications targetting XIlinx Alveo U280 and U50 devices. Makefile will be added in future to support commandline based implementation. 
Kernel files are named as *_kernel.cpp.

each application folder contains configuration files named as *\*.cfg* and necessary placement and memory port constraints are provided there. 
you can set constriants in vitis GUI flow as *--config \*.cfg* in the GUI command box of the binray container and kernels.

#### Performance comparison of Xilinx Accelaration Cards with Nvidia V100 GPU
Following Experimental Results shows the suitability of the FPGAs for Implicit applications involing smaller to medium sizes meshes to get better performance and energy saving. Higher accuracy of the model prediction shows that arithmetic units are always utilied as intended. 

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




