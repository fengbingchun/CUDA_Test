# CUDA_Test
**The main role of the project:**
- CUDA'usage(each test code gives the implementation of C ++ and CUDA, respectively, and gives the calculation time for each method)
- TensorRT's usage
- Simd's usage [Simd GitHub](https://github.com/ermig1979/Simd)
- OpenMP's usage
- SIMD's usage
- Assembly Language's usage(MASM, NASM)
- Eigen's usage

**CUDA test code(Note: depend on opencv):**
- simple
	- vector add: C = A + B
	- matrix multiplication: C = A * B
	- dot product
	- Julia
	- ripple
	- green ball
	- ray tracking
	- heat conduction
	- calculate histogram
	- streams' usage
- layer(approximate)
	- channel normalize(mean/standard deviation)
	- reverse
	- prior_vbox
- image process
	- bgr to gray
	- bgr to bgr565
	- gray image histogram equalization(only C++ implementation)
	- gray image edge detection: Laplacian(only C++ implementation)

**Eigen test code:**
- transpose
- determinant
- inverse matrix
- norm
- eigenvalues/eigenvectors
- SVD(Singular Value Decomposition)
- pseudoinverse
- trace
- mean, variance, standard deviation
- covariance matrix

**TensorRT 2.1.2 test code(only support linux):**
- MNIST
- MNIST API(use api produce network)
- GoogleNet
- CharRNN
- Plugin(add a custom layer)
- MNIST Infer(serialize TensorRT model)

**The project support platform:**
- windows10 64 bits: It can be directly build with VS2022 in windows10 64bits(Except TensorRT).
- Linux:
	- CUDA supports cmake build(file position: prj/linux_cuda_cmake)
	- TensorRT support cmake build(file position: prj/linux_tensorrt_cmake)
	- Simd_Test support cmake build(file position: prj/linux_simd_cmake)
	- AssemblyLanguage_Test support cmake build(file position: prj/linux_assembly_language_cmake)
	- Eigen_Test support cmake build(file position: prj/linux_eigen_cmake)

**Screenshot:**  
![](https://github.com/fengbingchun/CUDA_Test/blob/master/prj/x86_x64_vc12/Screenshot.png)

**Blog:** [fengbingchun](http://blog.csdn.net/fengbingchun/article/category/1531463)
