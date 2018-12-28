# Parallel-N-Queens

We designed a sequential program to solve the N-Queens problem recursively. Then MPI and CUDA are applied to accelerate it. C++ is used in this assignment.

A work of  
Fengkai Liu(fengkail@student.unimelb.edu.au)  
and  
Sheng Tang(shengt2@student.unimelb.edu.au)    
# Performance Evaluation
## Execution times for sequential implementation
Size | 8|9|10|11|12|13|14|15|16
---|---|---|---|---|---|---|---|---|---|
Time(seconds)|0.001|0.004|0.016|0.073|0.401|2.448|16.191|115.911|839.930|  

Based on the obtained rate of increase, an estimation of computation time for a larger size can be made, which reveals that it requires 2545 days or 7 years to calculate the solution for 22-Queens problem.

## Execution times for MPI implementation
The scalability of the MPI implementation has been tested in a range of parallelism settings from 2 processors to 32 processors with an Intel Core i5-6300U processor.

Size | 8|9|10|11|12|13|14|15|16
---|---|---|---|---|---|---|---|---|---|
nq=2|0.122|0.123|0.131|0.199|0.537|2.775|17.986|130.382|985.94|  
nq=8|0.326|0.348|0.338|0.378|0.479|1.272|6.453|44.454|322.97|  
nq=16|0.675|0.663|0.676|0.682|0.853|1.584|6.763|44.634|326.32|  
nq=32|1.683|1.775|1.854|1.784|1.943|2.668|7.920|46.28|328.84|  

## Execution times for CUDA implementation
In CUDA part, the time cost can separate into two parts. The time cost of malloc linear memories on the device. And the CUDA multi-thread processing part. we test it in a GeForce GTX 1070 Graphics card  

Size |12|13|14|15|16
---|---|---|---|---|---|
malloc|0.216|0.418|0.424|0.426|0.47|
multithread|0.035|0.457|1.123|3.942|18.511|
reduction|0|0|0|0|0|
sum|0.251|0.875|1.547|4.368|18.981|


---
For more N-queens result
https://oeis.org/A000170
