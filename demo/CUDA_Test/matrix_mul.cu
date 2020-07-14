#include "funset.hpp"
#include <iostream>
#include <cuda_runtime.h> // For the CUDA runtime routines (prefixed with "cuda_")
#include <device_launch_parameters.h>
#include "common.hpp"

// reference: C:\ProgramData\NVIDIA Corporation\CUDA Samples\v8.0\0_Simple\matrixMul
/* __global__: 函数类型限定符;在设备上运行;在主机端调用,计算能力3.2及以上可以在
设备端调用;声明的函数的返回值必须是void类型;对此类型函数的调用是异步的,即在
设备完全完成它的运行之前就返回了;对此类型函数的调用必须指定执行配置,即用于在
设备上执行函数时的grid和block的维度,以及相关的流(即插入<<<   >>>运算符);
a kernel,表示此函数为内核函数(运行在GPU上的CUDA并行计算函数称为kernel(内核函
数),内核函数必须通过__global__函数类型限定符定义);*/
template <int BLOCK_SIZE>
__global__ static void matrix_mul(const float* A, const float* B, float* C, int wA, int wB)
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
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;
	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Index of the first sub-matrix of A processed by the block
	int aBegin = wA * BLOCK_SIZE * by;
	// Index of the last sub-matrix of A processed by the block
	int aEnd = aBegin + wA - 1;
	// Step size used to iterate through the sub-matrices of A
	int aStep = BLOCK_SIZE;
	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;
	// Step size used to iterate through the sub-matrices of B
	int bStep = BLOCK_SIZE * wB;
	// Csub is used to store the element of the block sub-matrix that is computed by the thread
	float Csub = 0;

	// Loop over all the sub-matrices of A and B required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
		/* __shared__: 变量类型限定符；使用__shared__限定符，或者与__device__限
		定符连用，此时声明的变量位于block中的共享存储器空间中，与block具有相同
		的生命周期，仅可通过block内的所有线程访问；__shared__和__constant__变量
		默认为是静态存储；在__shared__前可以加extern关键字，但表示的是变量大小
		由执行参数确定；__shared__变量在声明时不能初始化；可以将CUDA C的关键字
		__shared__添加到变量声明中，这将使这个变量驻留在共享内存中；CUDA C编译
		器对共享内存中的变量与普通变量将分别采取不同的处理方式 */
		// Declaration of the shared memory array As used to store the sub-matrix of A
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		// Declaration of the shared memory array Bs used to store the sub-matrix of B
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		// Load the matrices from device memory to shared memory; each thread loads one element of each matrix
		As[ty][tx] = A[a + wA * ty + tx];
		Bs[ty][tx] = B[b + wB * ty + tx];

		/* __syncthreads: 对线程块中的线程进行同步；CUDA架构将确保，除非线程块
		中的每个线程都执行了__syncthreads()，否则没有任何线程能执行
		__syncthreads()之后的指令;在同一个block中的线程通过共享存储器(shared
		memory)交换数据，并通过栅栏同步(可以在kernel函数中需要同步的位置调用
		__syncthreads()函数)保证线程间能够正确地共享数据；使用clock()函数计时，
		在内核函数中要测量的一段代码的开始和结束的位置分别调用一次clock()函数，
		并将结果记录下来。由于调用__syncthreads()函数后，一个block中的所有
		thread需要的时间是相同的，因此只需要记录每个block执行需要的时间就行了，
		而不需要记录每个thread的时间 */
		// Synchronize to make sure the matrices are loaded
		__syncthreads();

		/* reference:
			https://devblogs.nvidia.com/parallelforall/new-compiler-features-cuda-8/
			https://stackoverflow.com/questions/22278631/what-does-pragma-unroll-do-exactly-does-it-affect-the-number-of-threads/22279341
		编译器默认情况下将循环展开小的次数，#pragma unroll能够指定循环
		以多少次展开(程序员必须保证按这个展开是正确的)，pragma unroll 后
		必须紧接着处理的循环，可选择在其后接一个数字，指定必须展开多少次循环，
		#pragma unroll 1 表示禁止编译器将循环展开。如果没指定次数，对于常数
		次的循环，循环将完全展开，对于不确定次数的循环，循环将不会展开。
		*/
#pragma unroll
		// Multiply the two matrices together; each thread computes one element of the block sub-matrix
		for (int k = 0; k < BLOCK_SIZE; ++k) {
			Csub += As[ty][k] * Bs[k][tx];
		}

		// Synchronize to make sure that the preceding computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}

	// Write the block sub-matrix to device memory; each thread writes one element
	int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	C[c + wB * ty + tx] = Csub;
}

__global__ static void matrix_mul(const float* A, const float* B, float* C, int colsA, int rowsA, int colsB, int rowsB)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float sum{ 0.f };
	for (int t = 0; t < colsA; ++t) {
		sum += A[y * colsA + t] * B[t * colsB + x];
	}

	C[offset] = sum;
}

int matrix_mul_gpu(const float* A, const float* B, float* C, int colsA, int rowsA, int colsB, int rowsB, float* elapsed_time)
{
	CHECK(colsA == rowsB);

	/* cudaEvent_t: CUDA event types，结构体类型, CUDA事件，用于测量GPU在某
	个任务上花费的时间，CUDA中的事件本质上是一个GPU时间戳，由于CUDA事件是在
	GPU上实现的，因此它们不适于对同时包含设备代码和主机代码的混合代码计时*/
	cudaEvent_t start, stop;
	// cudaEventCreate: 创建一个事件对象，异步启动
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// cudaEventRecord: 记录一个事件，异步启动,start记录起始时间
	cudaEventRecord(start, 0);

	size_t lengthA{ colsA * rowsA * sizeof(float) }, lengthB{ colsB * rowsB * sizeof(float) };
	size_t lengthC{ rowsA * colsB * sizeof(float) };
	float *d_A{ nullptr }, *d_B{ nullptr }, *d_C{ nullptr };

	// cudaMalloc: 在设备端分配内存
	cudaMalloc(&d_A, lengthA);
	cudaMalloc(&d_B, lengthB);
	cudaMalloc(&d_C, lengthC);

	/* cudaMemcpy: 在主机端和设备端拷贝数据,此函数第四个参数仅能是下面之一:
	(1). cudaMemcpyHostToHost: 拷贝数据从主机端到主机端
	(2). cudaMemcpyHostToDevice: 拷贝数据从主机端到设备端
	(3). cudaMemcpyDeviceToHost: 拷贝数据从设备端到主机端
	(4). cudaMemcpyDeviceToDevice: 拷贝数据从设备端到设备端
	(5). cudaMemcpyDefault: 从指针值自动推断拷贝数据方向,需要支持
	统一虚拟寻址(CUDA6.0及以上版本)
	cudaMemcpy函数对于主机是同步的 */
	cudaMemcpy(d_A, A, lengthA, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, lengthB, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_C, C, lengthC, cudaMemcpyHostToDevice);

	const int block_size{ 32 };
	/* dim3: 基于uint3定义的内置矢量类型，相当于由3个unsigned int类型组成的
	结构体，可表示一个三维数组，在定义dim3类型变量时，凡是没有赋值的元素都
	会被赋予默认值1 */
	dim3 dimsA(colsA, rowsA, 1);
	dim3 dimsB(colsB, rowsB, 1);
	CHECK(dimsA.x == dimsB.y);
	//fprintf(stderr, "MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);

	dim3 threads(block_size, block_size);
	dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

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
	matrix_mul<block_size> <<< grid, threads >>>(d_A, d_B, d_C, dimsA.x, dimsB.x); // 运行较快
	//matrix_mul<< < grid, threads >> >(d_A, d_B, d_C, colsA, rowsA, colsB, rowsB);

	/* cudaDeviceSynchronize: kernel的启动是异步的, 为了定位它是否出错, 一
	般需要加上cudaDeviceSynchronize函数进行同步; 将会一直处于阻塞状态，直到
	前面所有请求的任务已经被全部执行完毕，如果前面执行的某个任务失败，将会
	返回一个错误；当程序中有多个流，并且流之间在某一点需要通信时，那就必须
	在这一点处加上同步的语句，即cudaDeviceSynchronize；异步启动
	reference: https://stackoverflow.com/questions/11888772/when-to-call-cudadevicesynchronize */
	//cudaDeviceSynchronize();

	cudaMemcpy(C, d_C, lengthC, cudaMemcpyDeviceToHost);
	// cudaFree: 释放设备上由cudaMalloc函数分配的内存
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// cudaEventRecord: 记录一个事件，异步启动,stop记录结束时间
	cudaEventRecord(stop, 0);
	// cudaEventSynchronize: 事件同步，等待一个事件完成，异步启动
	cudaEventSynchronize(stop);
	// cudaEventElapseTime: 计算两个事件之间经历的时间，单位为毫秒，异步启动
	cudaEventElapsedTime(elapsed_time, start, stop);
	// cudaEventDestroy: 销毁事件对象，异步启动
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}

