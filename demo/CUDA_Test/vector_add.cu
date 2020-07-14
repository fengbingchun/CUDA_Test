#include "funset.hpp"
#include <iostream>
#include <cuda_runtime.h> // For the CUDA runtime routines (prefixed with "cuda_")
#include <device_launch_parameters.h>
#include "common.hpp"

// reference: C:\ProgramData\NVIDIA Corporation\CUDA Samples\v8.0\0_Simple\vectorAdd
/* __global__: 函数类型限定符;在设备上运行;在主机端调用,计算能力3.2及以上可以在
设备端调用;声明的函数的返回值必须是void类型;对此类型函数的调用是异步的,即在
设备完全完成它的运行之前就返回了;对此类型函数的调用必须指定执行配置,即用于在
设备上执行函数时的grid和block的维度,以及相关的流(即插入<<<   >>>运算符);
a kernel,表示此函数为内核函数(运行在GPU上的CUDA并行计算函数称为kernel(内核函
数),内核函数必须通过__global__函数类型限定符定义);*/
__global__ static void vector_add(const float *A, const float *B, float *C, int numElements)
{
	/* gridDim: 内置变量,用于描述线程网格的维度,对于所有线程块来说,这个
	变量是一个常数,用来保存线程格每一维的大小,即每个线程格中线程块的数量.
	一个grid为三维,为dim3类型；
	blockDim: 内置变量,用于说明每个block的维度与尺寸.为dim3类型,包含
	了block在三个维度上的尺寸信息;对于所有线程块来说,这个变量是一个常数,
	保存的是线程块中每一维的线程数量;
	blockIdx: 内置变量,变量中包含的值就是当前执行设备代码的线程块的索引;用
	于说明当前thread所在的block在整个grid中的位置,blockIdx.x取值范围是
	[0,gridDim.x-1],blockIdx.y取值范围是[0, gridDim.y-1].为uint3类型,
	包含了一个block在grid中各个维度上的索引信息;
	threadIdx: 内置变量,变量中包含的值就是当前执行设备代码的线程索引;用于
	说明当前thread在block中的位置;如果线程是一维的可获取threadIdx.x,如果
	是二维的还可获取threadIdx.y,如果是三维的还可获取threadIdx.z;为uint3类
	型,包含了一个thread在block中各个维度的索引信息 */
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numElements) {
		C[i] = A[i] + B[i];
	}
}

int vector_add_gpu(const float* A, const float* B, float* C, int numElements, float* elapsed_time)
{
	/* Error code to check return values for CUDA calls
	cudaError_t: CUDA Error types, 枚举类型,CUDA错误码,成功返回
	cudaSuccess(0),否则返回其它(>0) */
	cudaError_t err{ cudaSuccess };

	/* cudaEvent_t: CUDA event types，结构体类型, CUDA事件，用于测量GPU在某
	个任务上花费的时间，CUDA中的事件本质上是一个GPU时间戳，由于CUDA事件是在
	GPU上实现的，因此它们不适于对同时包含设备代码和主机代码的混合代码计时*/
	cudaEvent_t start, stop;
	// cudaEventCreate: 创建一个事件对象，异步启动
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// cudaEventRecord: 记录一个事件，异步启动,start记录起始时间
	cudaEventRecord(start, 0);

	size_t length{ numElements * sizeof(float) };
	float *d_A{ nullptr }, *d_B{ nullptr }, *d_C{ nullptr };

	// cudaMalloc: 在设备端分配内存
	err = cudaMalloc(&d_A, length);
	if (err != cudaSuccess) {
		// cudaGetErrorString: 返回错误码的描述字符串
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
			cudaGetErrorString(err));
		return -1;
	}
	err = cudaMalloc(&d_B, length);
	if (err != cudaSuccess) PRINT_ERROR_INFO(cudaMalloc);
	err = cudaMalloc(&d_C, length);
	if (err != cudaSuccess) PRINT_ERROR_INFO(cudaMalloc);

	/* cudaMemcpy: 在主机端和设备端拷贝数据,此函数第四个参数仅能是下面之一:
	(1). cudaMemcpyHostToHost: 拷贝数据从主机端到主机端
	(2). cudaMemcpyHostToDevice: 拷贝数据从主机端到设备端
	(3). cudaMemcpyDeviceToHost: 拷贝数据从设备端到主机端
	(4). cudaMemcpyDeviceToDevice: 拷贝数据从设备端到设备端
	(5). cudaMemcpyDefault: 从指针值自动推断拷贝数据方向,需要支持
	统一虚拟寻址(CUDA6.0及以上版本)
	cudaMemcpy函数对于主机是同步的 */
	err = cudaMemcpy(d_A, A, length, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) PRINT_ERROR_INFO(cudaMemcpy);
	err = cudaMemcpy(d_B, B, length, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) PRINT_ERROR_INFO(cudaMemcpy);

	// Launch the Vector Add CUDA kernel
	const int threadsPerBlock{ 256 };
	const int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	fprintf(stderr, "CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	/* <<< >>>: 为CUDA引入的运算符,指定线程网格和线程块维度等,传递执行参
	数给CUDA编译器和运行时系统,用于说明内核函数中的线程数量,以及线程是如何
	组织的;尖括号中这些参数并不是传递给设备代码的参数,而是告诉运行时如何
	启动设备代码,传递给设备代码本身的参数是放在圆括号中传递的,就像标准的函
	数调用一样;不同计算能力的设备对线程的总数和组织方式有不同的约束;必须
	先为kernel中用到的数组或变量分配好足够的空间,再调用kernel函数,否则在
	GPU计算时会发生错误,例如越界等;
	使用运行时API时,需要在调用的内核函数名与参数列表直接以<<<Dg,Db,Ns,S>>>
	的形式设置执行配置,其中：Dg是一个dim3型变量,用于设置grid的维度和各个
	维度上的尺寸.设置好Dg后,grid中将有Dg.x*Dg.y*Dg.z个block;Db是
	一个dim3型变量,用于设置block的维度和各个维度上的尺寸.设置好Db后,每个
	block中将有Db.x*Db.y*Db.z个thread;Ns是一个size_t型变量,指定各块为此调
	用动态分配的共享存储器大小,这些动态分配的存储器可供声明为外部数组
	(extern __shared__)的其他任何变量使用;Ns是一个可选参数,默认值为0;S为
	cudaStream_t类型,用于设置与内核函数关联的流.S是一个可选参数,默认值0. */
	vector_add << <blocksPerGrid, threadsPerBlock >> >(d_A, d_B, d_C, numElements);
	/* cudaGetLastError: 在同一个主机线程中,返回运行时调用中产生的最后一个
	错误并将其重置为cudaSuccess;此函数也可能返回以前异步启动的错误码;当有
	多个错误在对cudaGetLastError的调用之间发生时,仅最后一个错误会被报告;
	kernel的启动是异步的,为了定位它是否出错,一般需要加上
	cudaDeviceSynchronize函数进行同步,然后再调用cudaGetLastError函数;*/
	err = cudaGetLastError();
	if (err != cudaSuccess) PRINT_ERROR_INFO(cudaGetLastError);
	// Copy the device result vector in device memory to the host result vector in host memory.
	err = cudaMemcpy(C, d_C, length, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) PRINT_ERROR_INFO(cudaMemcpy);

	// cudaFree: 释放设备上由cudaMalloc函数分配的内存
	err = cudaFree(d_A);
	if (err != cudaSuccess) PRINT_ERROR_INFO(cudaFree);
	err = cudaFree(d_B);
	if (err != cudaSuccess) PRINT_ERROR_INFO(cudaFree);
	err = cudaFree(d_C);
	if (err != cudaSuccess) PRINT_ERROR_INFO(cudaFree);

	// cudaEventRecord: 记录一个事件，异步启动,stop记录结束时间
	cudaEventRecord(stop, 0);
	// cudaEventSynchronize: 事件同步，等待一个事件完成，异步启动
	cudaEventSynchronize(stop);
	// cudaEventElapseTime: 计算两个事件之间经历的时间，单位为毫秒，异步启动
	cudaEventElapsedTime(elapsed_time, start, stop);
	// cudaEventDestroy: 销毁事件对象，异步启动
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return err;
}
