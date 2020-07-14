#include "funset.hpp"
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "common.hpp"

namespace {
/* __global__: 函数类型限定符;在设备上运行;在主机端调用,计算能力3.2及以上可以在
设备端调用;声明的函数的返回值必须是void类型;对此类型函数的调用是异步的,即在
设备完全完成它的运行之前就返回了;对此类型函数的调用必须指定执行配置,即用于在
设备上执行函数时的grid和block的维度,以及相关的流(即插入<<<   >>>运算符);
a kernel,表示此函数为内核函数(运行在GPU上的CUDA并行计算函数称为kernel(内核函
数),内核函数必须通过__global__函数类型限定符定义);*/
__global__ void histogram(const unsigned char* src, int length, int* dst)
{
	/* __shared__: 变量类型限定符；使用__shared__限定符，或者与__device__限
	定符连用，此时声明的变量位于block中的共享存储器空间中，与block具有相同
	的生命周期，仅可通过block内的所有线程访问；__shared__和__constant__变量
	默认为是静态存储；在__shared__前可以加extern关键字，但表示的是变量大小
	由执行参数确定；__shared__变量在声明时不能初始化；可以将CUDA C的关键字
	__shared__添加到变量声明中，这将使这个变量驻留在共享内存中；CUDA C编译
	器对共享内存中的变量与普通变量将分别采取不同的处理方式 */
	// clear out the accumulation buffer called temp since we are launched with
	// 256 threads, it is easy to clear that memory with one write per thread
	__shared__ int temp[256]; // 共享内存缓冲区
	temp[threadIdx.x] = 0;
	/* __syncthreads: 对线程块中的线程进行同步；CUDA架构将确保，除非线程块
	中的每个线程都执行了__syncthreads()，否则没有任何线程能执行
	__syncthreads()之后的指令;在同一个block中的线程通过共享存储器(shared
	memory)交换数据，并通过栅栏同步(可以在kernel函数中需要同步的位置调用
	__syncthreads()函数)保证线程间能够正确地共享数据；使用clock()函数计时，
	在内核函数中要测量的一段代码的开始和结束的位置分别调用一次clock()函数，
	并将结果记录下来。由于调用__syncthreads()函数后，一个block中的所有
	thread需要的时间是相同的，因此只需要记录每个block执行需要的时间就行了，
	而不需要记录每个thread的时间 */
	__syncthreads();

	/* gridDim: 内置变量,用于描述线程网格的维度,对于所有线程块来说,这个
	变量是一个常数,用来保存线程格每一维的大小,即每个线程格中线程块的数量.
	为dim3类型；
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
	// calculate the starting index and the offset to the next block that each thread will be processing
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	while (i < length) {
		/* atomicAdd: 原子操作,底层硬件将确保当执行这些原子操作时，其
		它任何线程都不会读取或写入地址addr上的值。原子函数(atomic
		function)对位于全局或共享存储器的一个32位或64位字执行
		read-modify-write的原子操作。也就是说，当多个线程同时访问全局或
		共享存储器的同一位置时，保证每个线程能够实现对共享可写数据的互
		斥操作：在一个操作完成之前，其它任何线程都无法访问此地址。之所
		以将这一过程称为原子操作，是因为每个线程的操作都不会影响到其它
		线程。换句话说，原子操作能够保证对一个地址的当前操作完成之前，
		其它线程都不能访问这个地址。
		atomicAdd(addr,y)：将生成一个原子的操作序列，这个操作序列包括读
		取地址addr处的值，将y增加到这个值，以及将结果保存回地址addr。 */
		atomicAdd(&temp[src[i]], 1);
		i += stride;
	}

	// sync the data from the above writes to shared memory then add the shared memory values to the values from
	// the other thread blocks using global memory atomic adds same as before, since we have 256 threads,
	// updating the global histogram is just one write per thread!
	__syncthreads();
	// 将每个线程块的直方图合并为单个最终的直方图
	atomicAdd(&(dst[threadIdx.x]), temp[threadIdx.x]);
}

__global__ void equalization(const unsigned char* src, int length, unsigned char* dst)
{

}

} // namespace

int histogram_equalization_gpu(const unsigned char* src, int width, int height, unsigned char* dst, float* elapsed_time)
{
	const int hist_sz{ 256 }, length{ width * height }, byte_sz{ (int)sizeof(unsigned char) * length};
	unsigned char *dev_src{ nullptr }, *dev_dst{ nullptr };
	int* dev_hist{ nullptr };

	// cudaMalloc: 在设备端分配内存
	cudaMalloc(&dev_src, byte_sz);
	cudaMalloc(&dev_dst, byte_sz);
	cudaMalloc(&dev_hist, hist_sz * sizeof(int));
	/* cudaMemcpy: 在主机端和设备端拷贝数据,此函数第四个参数仅能是下面之一:
	(1). cudaMemcpyHostToHost: 拷贝数据从主机端到主机端
	(2). cudaMemcpyHostToDevice: 拷贝数据从主机端到设备端
	(3). cudaMemcpyDeviceToHost: 拷贝数据从设备端到主机端
	(4). cudaMemcpyDeviceToDevice: 拷贝数据从设备端到设备端
	(5). cudaMemcpyDefault: 从指针值自动推断拷贝数据方向,需要支持
	统一虚拟寻址(CUDA6.0及以上版本)
	cudaMemcpy函数对于主机是同步的 */
	cudaMemcpy(dev_src, src, byte_sz, cudaMemcpyHostToDevice);

	/* cudaMemset: 存储器初始化函数,在GPU内存上执行。用指定的值初始化或设置
	设备内存 */
	cudaMemset(dev_hist, 0, hist_sz * sizeof(int));

	// cudaDeviceProp: cuda设备属性结构体
	// kernel launch - 2x the number of mps gave best timing
	cudaDeviceProp prop;
	// cudaGetDeviceProperties: 获取GPU设备相关信息
	cudaGetDeviceProperties(&prop, 0);
	// cudaDeviceProp::multiProcessorCount: 设备上多处理器的数量
	int blocks = prop.multiProcessorCount;

	TIME_START_GPU

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
	block中将有Db.x*Db.y*Db.z个thread;Ns是一个unsigned int型变量,指定各块为此调
	用动态分配的共享存储器大小,这些动态分配的存储器可供声明为外部数组
	(extern __shared__)的其他任何变量使用;Ns是一个可选参数,默认值为0;S为
	cudaStream_t类型,用于设置与内核函数关联的流.S是一个可选参数,默认值0. */
	// 当线程块的数量为GPU中处理器数量的2倍时，将达到最优性能
	// Note: 核函数不支持传入参数为vector的data()指针，需要cudaMalloc和cudaMemcpy，因为vector是在主机内存中
	histogram << <blocks * 2, 256 >> >(dev_src, length, dev_hist);

	TIME_END_GPU

	cudaMemcpy(dst, dev_dst, byte_sz, cudaMemcpyDeviceToHost);

	// cudaFree: 释放设备上由cudaMalloc函数分配的内存
	cudaFree(dev_src);
	cudaFree(dev_hist);
	cudaFree(dev_dst);

	return 0;
}

