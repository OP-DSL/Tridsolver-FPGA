Range of HLS optimisation we have applied inorder to get expected performance for Batched thomas solver Library as well as applications we have implemented. All the library and application are developed using Vivado C++ and we guides the compiler using HLS Pragmas. Following are the main optimisations in this work which could be useful for other works. 

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
As inner loop have loop carried dependency, this limites Initiation Interval(II or g) of the inner loop close to 30 for FP32 arithmetic and close to 60 for FP64 arithmetic.  This will lead to higher latency like Batch_size*N*30 for FP32 and Batch_size*N*60 for FP64. Inorder to improve the performance, we need to tatget ideal II=1 for inner loop. Here we note that dependecy distance is 1. 

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
above loop transformation of solving systems in interleaved manner increases the dendency distance of the most inner loop to g. This makes consecutive iterations to be executed each clock cycles and reaching Ideal II=1.

### Overcoming Limited number of Memory Ports
Both of above nested loops will hit limited mmeory port issue. This is becuase, there are two load operation on memory b and one write operation but BRAM natively support two ports where only two memory operations can be done in parallel. Due to this limitation itroducing a temporary storage to store dependency distance number of elements will allow us to achive II=1 without replicating the memory. This is as follows
```C
for(int j =0; j < Batch_size/g; j++){
  for(int i = 1; i < N; i++){
    for(int k = 0; k < g; k++){
      #pragma HLS dependence variable=b_last RAW distance=DIST true
      #pragma HLS dependence variable=d_last RAW distance=DIST true
      int ind = (j*g+k)*N+i; 
      float w = a[ind] / b_last[g];
      b[ind] = b[ind] - w * c[ind-g];
      d[ind] = d[ind] - w * d_last[g];
      b_last[g] = b[ind];
      d_last[g] = d[ind];
    }
   }
}
```
Here we guide compiler about the dependency distance(DIST=g) using #pragma HLS dependence directive. if g is a constant, compiler will automatically find the dependency distance. 

### Ping Pong buffers inference using Nested loop Vs Flatened Loop 
Since forward loop memory access and backward loop memory accesss are in opposite direction in Thomas solver, we need to transfer data between two loops using memory. When we target higher perfromance, each loops should execute in parallel. Basically they are mapped to two hardware modules which operates in parallel. Inorder to infer that, memories used in those loops should have one operations, first loop writes the data and second loop reads the data. This is called as ping pong buffers or double buffers, requires twice memory as executing one loop after another. We can use the Xilinx data flow directive to execute both loops in parallel. 

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

For the above implmentation compiler will automatically implement the ping pong buffers. But one disadvantage of nested loops in ping pong buffer synchronization will include arithmetic pipeline latency as well. Total number of clock cycles to swap the location will be clocks to read/write all the values plus arithmetic pipeline latency. This overhead become significant when Batch size is huge. We eliminate this overhead by manually implementing the ping pong buffers by partitioning the memory and implmenting this in a flattened loop as follows. 


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

### HBM FIFO 
