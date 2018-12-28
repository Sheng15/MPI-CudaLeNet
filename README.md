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
The scalability of the MPI implementation has been tested in a range of parallelism settings from 2 processors to 32 processors.

Size | 8|9|10|11|12|13|14|15|16
---|---|---|---|---|---|---|---|---|---|
nq=2|0.122|0.123|0.131|0.199|0.537|2.775|17.986|130.382|985.94|  
---|---|---|---|---|---|---|---|---|---|
nq=4|0.172|0.180|0.175|0.210|0.361|1.314|7.615|53.777|411.91|  
---|---|---|---|---|---|---|---|---|---|
nq=8|0.326|0.348|0.338|0.378|0.479|1.272|6.453|44.454|322.97|  
---|---|---|---|---|---|---|---|---|---|
nq=16|0.675|0.663|0.676|0.682|0.853|1.584|6.763|44.634|326.32|  
---|---|---|---|---|---|---|---|---|---|
nq=32|1.683|1.775|1.854|1.784|1.943|2.668|7.920|46.28|328.84|  


---
For more N-queens result
https://oeis.org/A000170
