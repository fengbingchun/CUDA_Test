#include "funset.hpp"
#include <iostream>
#include <algorithm>
#include <memory>
#include <vector>
#include <cuda_runtime.h> // For the CUDA runtime routines (prefixed with "cuda_")
#include <device_launch_parameters.h>
#include "common.hpp"

/* __global__: 函数类型限定符;在设备上运行;在主机端调用,计算能力3.2及以上可以在
设备端调用;声明的函数的返回值必须是void类型;对此类型函数的调用是异步的,即在
设备完全完成它的运行之前就返回了;对此类型函数的调用必须指定执行配置,即用于在
设备上执行函数时的grid和block的维度,以及相关的流(即插入<<<   >>>运算符);
a kernel,表示此函数为内核函数(运行在GPU上的CUDA并行计算函数称为kernel(内核函
数),内核函数必须通过__global__函数类型限定符定义); */
__global__ static void stream_kernel(int* a, int* b, int* c, int length)
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
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < length) {
		int idx1 = (idx + 1) % 256;
		int idx2 = (idx + 2) % 256;
		float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
		float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
		c[idx] = (as + bs) / 2;
	}
}

int streams_gpu_1(const int* a, const int* b, int* c, int length, float* elapsed_time)
{
	// cudaDeviceProp: cuda设备属性结构体
	cudaDeviceProp prop;
	// cudaGetDeviceProperties: 获取GPU设备相关信息
	cudaGetDeviceProperties(&prop, 0);
	/* cudaDeviceProp::deviceOverlap: GPU是否支持设备重叠(Device Overlap)功
	能,支持设备重叠功能的GPU能够在执行一个CUDA C核函数的同时，还能在设备与
	主机之间执行复制等操作 */
	if (!prop.deviceOverlap) {
		printf("Device will not handle overlaps, so no speed up from streams\n");
		return -1;
	}

	/* cudaEvent_t: CUDA event types,结构体类型, CUDA事件,用于测量GPU在某
	个任务上花费的时间,CUDA中的事件本质上是一个GPU时间戳,由于CUDA事件是在
	GPU上实现的,因此它们不适于对同时包含设备代码和主机代码的混合代码计时 */
	cudaEvent_t start, stop;
	// cudaEventCreate: 创建一个事件对象,异步启动
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// cudaEventRecord: 记录一个事件,异步启动,start记录起始时间
	cudaEventRecord(start, 0);

	/* cudaStream_t: cuda 流，结构体类型, CUDA流表示一个GPU操作队列，并且该
	队列中的操作将以指定的顺序执行。可以将每个流视为GPU上的一个任务，并且这
	些任务可以并行执行。 */
	cudaStream_t stream;
	// cudaStreamCreate: 初始化流，创建一个新的异步流
	cudaStreamCreate(&stream);

	int *host_a{ nullptr }, *host_b{ nullptr }, *host_c{ nullptr };
	int *dev_a{ nullptr }, *dev_b{ nullptr }, *dev_c{ nullptr };
	const int N{ length / 20 };

	// cudaMalloc: 在设备端分配内存
	cudaMalloc(&dev_a, N * sizeof(int));
	cudaMalloc(&dev_b, N * sizeof(int));
	cudaMalloc(&dev_c, N * sizeof(int));
	/* cudaHostAlloc: 分配主机内存(固定内存)。C库函数malloc将分配标准的，可
	分页的(Pagable)主机内存，而cudaHostAlloc将分配页锁定的主机内存。页锁定内
	存也称为固定内存(Pinned Memory)或者不可分页内存，它有一个重要的属性：操作系
	统将不会对这块内存分页并交换到磁盘上，从而确保了该内存始终驻留在物理内
	存中。因此，操作系统能够安全地使某个应用程序访问该内存的物理地址，因为
	这块内存将不会被破坏或者重新定位。由于GPU知道内存的物理地址，因此可以通
	过"直接内存访问(Direct Memory Access, DMA)"技术来在GPU和主机之间复制数据。
	固定内存是一把双刃剑。当使用固定内存时，你将失去虚拟内存的所有功能。
	建议：仅对cudaMemcpy调用中的源内存或者目标内存，才使用页锁定内存，并且在
	不再需要使用它们时立即释放。 */
	// 分配由流使用的页锁定内存
	cudaHostAlloc(&host_a, length * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc(&host_b, length * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc(&host_c, length * sizeof(int), cudaHostAllocDefault);

	//for (int i = 0; i < length; ++i) {
	//	host_a[i] = a[i];
	//	host_b[i] = b[i];
	//}
	memcpy(host_a, a, length * sizeof(int));
	memcpy(host_b, b, length * sizeof(int));

	for (int i = 0; i < length; i += N) {
		/* cudaMemcpyAsync: 在GPU与主机之间复制数据。cudaMemcpy的行为类
		似于C库函数memcpy。尤其是，这个函数将以同步方式执行，这意味着，
		当函数返回时，复制操作就已经完成，并且在输出缓冲区中包含了复制
		进去的内容。异步函数的行为与同步函数相反，在调用cudaMemcpyAsync时，
		只是放置了一个请求，表示在流中执行一次内存复制操作，这个流是通过
		参数stream来指定的。当函数返回时，我们无法确保复制操作是否已经
		启动，更无法保证它们是否已经结束。我们能够得到的保证是，复制操作肯定
		会当下一个被放入流中的操作之前执行。任何传递给cudaMemcpyAsync的主机
		内存指针都必须已经通过cudaHostAlloc分配好内存。也就是，你只能以异步
		方式对页锁定内存进行复制操作 */
		// 将锁定内存以异步方式复制到设备上
		cudaMemcpyAsync(dev_a, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(dev_b, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);

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
		stream_kernel << <N / 256, 256, 0, stream >> >(dev_a, dev_b, dev_c, N);

		cudaMemcpyAsync(host_c + i, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost, stream);
	}

	/* cudaStreamSynchronize: 等待传入流中的操作完成，主机在继续执行之前，要
	等待GPU执行完成 */
	cudaStreamSynchronize(stream);

	//for (int i = 0; i < length; ++i)
	//	c[i] = host_c[i];
	memcpy(c, host_c, length * sizeof(int));

	// cudaFreeHost: 释放设备上由cudaHostAlloc函数分配的内存
	cudaFreeHost(host_a);
	cudaFreeHost(host_b);
	cudaFreeHost(host_c);
	// cudaFree: 释放设备上由cudaMalloc函数分配的内存
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	// cudaStreamDestroy: 销毁流
	cudaStreamDestroy(stream);

	// cudaEventRecord: 记录一个事件,异步启动,stop记录结束时间
	cudaEventRecord(stop, 0);
	// cudaEventSynchronize: 事件同步,等待一个事件完成,异步启动
	cudaEventSynchronize(stop);
	// cudaEventElapseTime: 计算两个事件之间经历的时间,单位为毫秒,异步启动
	cudaEventElapsedTime(elapsed_time, start, stop);
	// cudaEventDestroy: 销毁事件对象,异步启动
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}

int streams_gpu_2(const int* a, const int* b, int* c, int length, float* elapsed_time)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	if (!prop.deviceOverlap) {
		printf("Device will not handle overlaps, so no speed up from streams\n");
		return -1;
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cudaStream_t stream0, stream1;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);

	int *host_a{ nullptr }, *host_b{ nullptr }, *host_c{ nullptr };
	int *dev_a0{ nullptr }, *dev_b0{ nullptr }, *dev_c0{ nullptr };
	int *dev_a1{ nullptr }, *dev_b1{ nullptr }, *dev_c1{ nullptr };
	const int N{ length / 20 };

	cudaMalloc(&dev_a0, N * sizeof(int));
	cudaMalloc(&dev_b0, N * sizeof(int));
	cudaMalloc(&dev_c0, N * sizeof(int));
	cudaMalloc(&dev_a1, N * sizeof(int));
	cudaMalloc(&dev_b1, N * sizeof(int));
	cudaMalloc(&dev_c1, N * sizeof(int));
	cudaHostAlloc(&host_a, length * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc(&host_b, length * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc(&host_c, length * sizeof(int), cudaHostAllocDefault);

	memcpy(host_a, a, length * sizeof(int));
	memcpy(host_b, b, length * sizeof(int));

	for (int i = 0; i < length; i += N * 2) {
		//cudaMemcpyAsync(dev_a0, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0);
		//cudaMemcpyAsync(dev_b0, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0);
		//stream_kernel << <N / 256, 256, 0, stream0 >> >(dev_a0, dev_b0, dev_c0, N);
		//cudaMemcpyAsync(host_c + i, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream0);

		//cudaMemcpyAsync(dev_a1, host_a + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
		//cudaMemcpyAsync(dev_b1, host_b + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
		//stream_kernel << <N / 256, 256, 0, stream1 >> >(dev_a1, dev_b1, dev_c1, N);
		//cudaMemcpyAsync(host_c + i + N, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1);

		// 推荐采用宽度优先方式
		cudaMemcpyAsync(dev_a0, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(dev_a1, host_a + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);

		cudaMemcpyAsync(dev_b0, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(dev_b1, host_b + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);

		stream_kernel << <N / 256, 256, 0, stream0 >> >(dev_a0, dev_b0, dev_c0, N);
		stream_kernel << <N / 256, 256, 0, stream1 >> >(dev_a1, dev_b1, dev_c1, N);

		cudaMemcpyAsync(host_c + i, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream0);
		cudaMemcpyAsync(host_c + i + N, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1);
	}

	cudaStreamSynchronize(stream0);
	cudaStreamSynchronize(stream1);

	memcpy(c, host_c, length * sizeof(int));

	cudaFreeHost(host_a);
	cudaFreeHost(host_b);
	cudaFreeHost(host_c);
	cudaFree(dev_a0);
	cudaFree(dev_b0);
	cudaFree(dev_c0);
	cudaFree(dev_a1);
	cudaFree(dev_b1);
	cudaFree(dev_c1);
	cudaStreamDestroy(stream0);
	cudaStreamDestroy(stream1);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(elapsed_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}

int streams_gpu(const int* a, const int* b, int* c, int length, float* elapsed_time)
{
	int ret{ 0 };
	//ret = streams_gpu_1(a, b, c, length, elapsed_time); // 使用单个流
	ret = streams_gpu_2(a, b, c, length, elapsed_time); // 使用多个流

	return ret;
}
