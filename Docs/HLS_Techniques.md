Range of HLS optimisation we have applied inorder to get performance gain for Batched thomas solver Library as well as applications we have implemented. All the library and application are developed using Vivado C++ and we guides the compiler using HLS Pragmas. Following are the main optimisations which could be useful for other works. 

## Thomas Solver Optimisations

### Thomas solver II=1 
One of the main disadvantage of the Thomas algorithm is it's sequential execution due to loop carried dependency. Following nested loop do thomas forward solve for Batch_size number of Systems. This is a simple representation of batched forward solve where all arrays are onchip memories without considering the onchip memory limitation for larger batch size.  

```C
for(int j =0; j < Batch_size; j++){
  for(int i = 1; i < N; i++){
    int ind = j*N+i; 
    float w = a[ind] / b[ind-1];
    b[ind] = b[ind] - w * c[ind-1];
    d[ind] = d[ind] - w * d[ind-1];
   }
}
```
As inner loop have loop carried dependency, this limites Initiation Interval(II or g) of the inner loop close to 30 for FP32 arithmetic and close to 60 for FP64 arithmetic.  This will lead to higher latency like `Batch_size*N*30` for FP32 and `Batch_size*N*60` for FP64. Inorder to improve the performance, we need to tatget ideal II=1 for inner loop. Here we note that dependecy distance is 1. 

```C
for(int j =0; j < Batch_size/g; j++){
  for(int i = 1; i < N; i++){
    for(int k = 0; k < g; k++){
      int ind = (j*g+k)*N+i; 
      float w = a[ind] / b[ind-g];
      b[ind] = b[ind] - w * c[ind-g];
      d[ind] = d[ind] - w * d[ind-g];
    }
   }
}
```
above loop transformation of solving systems in interleaved manner increases the dendency distance of the most inner loop to `g`. This makes consecutive iterations to be executed each clock cycles and reaching Ideal II=1.

### Overcoming Limited number of Memory Ports
Both of above nested loops will hit limited mmeory port issue. This is becuase, there are two load operation on memory b and one write operation but BRAM natively support two ports where only two memory operations can be done in parallel. Due to this limitation itroducing a temporary storage to store dependency distance number of elements will allow us to achive II=1 without replicating the memory. This is as follows
```C
#pragma HLS dependence variable=b_last RAW distance=DIST true
#pragma HLS dependence variable=d_last RAW distance=DIST true
```
Here we guide compiler about the dependency distance(DIST=g) using `#pragma HLS dependence` directive. if g is a constant, compiler will automatically find the dependency distance. 

### Ping Pong buffers inference using Nested loop Vs Flatened Loop 
Since forward loop memory access and backward loop memory accesss are in opposite direction in Thomas solver, we need to transfer data between two loops using memory. When we target higher perfromance, each loops should execute in parallel. Basically they are mapped to two hardware modules which operates in parallel. Inorder to infer that, memories used in those loops should have one operations, first loop writes the data and second loop reads the data. This is called as ping pong buffers or double buffers, requires twice memory as executing one loop after another. We can use the Xilinx `dataflow` directive to execute both loops in parallel. 

```C
for(int j =0; j < Batch_size/g; j++){
  #pragma HLS dataflow
  for(int i = 1; i < N; i++){
    for(int k = 0; k < g; k++){
      // forward loop
    }
   }
   
  for(int i = 1; i < N; i++){
    for(int k = 0; k < g; k++){
      // backward loop
    }
  }
}
```  

For the above implmentation compiler will automatically infer the ping pong buffers for data movement. But, the disadvantage of nested loops in ping pong buffer synchronization will include arithmetic pipeline latency as well. Total number of clock cycles to swap the location will be clock cycles to read/write all the values plus arithmetic pipeline latency. This overhead become significant when Batch size is huge. We eliminate this overhead by manually implementing the ping pong buffers by partitioning the memory and implmenting this in a flattened loop as follows. 


```C
#pragma HLS dataflow
for(int j =0; j < Batch_size/g*N*g; j++){
  // forward loop
}
// both loops are communicating through FIFO stream
for(int j =0; j < Batch_size/g*N*g; j++){
  // backward loop
}
```

### Support for Streaming Interface 
Streaming interface gives the flexibility that any amount of data can be continuously transferrred, independent execution of the module, and doesn't require any complex controls. The above `C` code get's input from on chip memory and output interface also a memory. Inorder to support stream interface, we have to put `g` number of system in the on chip memory, requiring a separate loop we call as `thomas_interleave`.

### Timing Improvement 
One of the challenge when scaling to multile thomas solvers is achiving a good operating frequency.  Data type size of the loop control variables plays a important role in derterming the critical path latency. In our designs we use arbitary bit length integer data types from `ap_uint.h` to improve the critical path timing. 

### Vectorisation 
Vectorization of multiple solvers under same control logic improves device resource consumption although very large vector factors lead to longer critical paths. 
We use 256 bit data path(`ap_uint<256>`) which can support 8 floating point solvers or 4 double precession solvers. `union` data strcture was used to reinterpret N bit data path as double/float or vice versa. Alternatively C `struct` data type can be used to implement vectorised data path but we noticed Vitis 2019.2 compiler allocate separate BRAM for each struct element array.   

## Multi Dimensional ADI optimisations

### Caching to improve the strided memory access 
In a 3D mesh, acccesing along y lines and z lines are challenging as consecutive access locations are non continuous. In order to get better memory perfromance we use on chip memory as cache to support burst memory transfers. if we want to read y lines, we read entire XY plane first and then we extract y lines from on chip memory. Here also ping pong buffer technique is used to support parallel external memory access and y line extraction.

### Fusing Coefficient Generation and Thomas solver 


### HBM Delay Buffer 
Implementing a big FIFO will require huge on chip memory, that size sometimes not be availble in the target device. In that cases, We can use external memory as a Big FIFO/ Delay buffer. Challenge here is, we have to synchronize the read and write pointers to implement where there is read after write dependency.  

```C
for(int i =0; i < Transfer_size + Delay_size; i++){
  if(i < Transfer_size){
      ext_mem[i] = in_f.pop();
  }
  if(i >= Delay_size){
      out_f.push(ext_mem[i-Delay_size]);
  }
}
```
above code will implement functionally correct delay buffer but it will be very slow. This is because compiler will try to enforce the inter iteration dependency due to multi clock cycles latency in read and write of external memory. This latency is usually 100-200 clock cycles, which leaves the II of loop in higher value. We can get ideal II=1 if we know the Delay_size is greater than ~200 clcoks by providing dependency distance using a pragma directive as follows.  

```C
#pragma HLS dependence variable=ext_mem RAW distance=200 true
```
We note that this construct worked in Vitis 2019.2 and didn't work in Vitis 2020.x as well as 2021.x

