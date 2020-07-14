#include "funset.hpp"
#include <iostream>
#include <algorithm>
#include <memory>
#include <cuda_runtime.h> // For the CUDA runtime routines (prefixed with "cuda_")
#include <device_launch_parameters.h>
#include "common.hpp"

/* __global__: 函数类型限定符;在设备上运行;在主机端调用,计算能力3.2及以上可以在
设备端调用;声明的函数的返回值必须是void类型;对此类型函数的调用是异步的,即在
设备完全完成它的运行之前就返回了;对此类型函数的调用必须指定执行配置,即用于在
设备上执行函数时的grid和block的维度,以及相关的流(即插入<<<   >>>运算符);
a kernel,表示此函数为内核函数(运行在GPU上的CUDA并行计算函数称为kernel(内核函
数),内核函数必须通过__global__函数类型限定符定义);*/
__global__ static void dot_product(const float* A, const float* B, float* partial_C, int elements_num)
{
	/* __shared__: 变量类型限定符；使用__shared__限定符，或者与__device__限
	定符连用，此时声明的变量位于block中的共享存储器空间中，与block具有相同
	的生命周期，仅可通过block内的所有线程访问；__shared__和__constant__变量
	默认为是静态存储；在__shared__前可以加extern关键字，但表示的是变量大小
	由执行参数确定；__shared__变量在声明时不能初始化；可以将CUDA C的关键字
	__shared__添加到变量声明中，这将使这个变量驻留在共享内存中；CUDA C编译
	器对共享内存中的变量与普通变量将分别采取不同的处理方式 */
	__shared__ float cache[256]; // == threadsPerBlock

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
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	float tmp{ 0.f };
	while (tid < elements_num) {
		tmp += A[tid] * B[tid];
		tid += blockDim.x * gridDim.x;
	}

	// 设置cache中相应位置上的值
	// 共享内存缓存中的偏移就等于线程索引；线程块索引与这个偏移无关，因为每
	// 个线程块都拥有该共享内存的私有副本
	cache[cacheIndex] = tmp;

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

	// 对于规约运算来说，以下code要求threadPerBlock必须是2的指数
	int i = blockDim.x / 2;
	while (i != 0) {
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];

		// 在循环迭代中更新了共享内存变量cache，并且在循环的下一次迭代开始之前，
		// 需要确保当前迭代中所有线程的更新操作都已经完成
		__syncthreads();
		i /= 2;
	}

	// 只有cacheIndex == 0的线程执行这个保存操作，这是因为只有一个值写入到
	// 全局内存，因此只需要一个线程来执行这个操作，当然你也可以选择任何一个
	// 线程将cache[0]写入到全局内存
	if (cacheIndex == 0)
		partial_C[blockIdx.x] = cache[0];
}

static int dot_product_gpu_1(const float* A, const float* B, float* value, int elements_num, float* elapsed_time)
{
	/* cudaEvent_t: CUDA event types,结构体类型, CUDA事件,用于测量GPU在某
	个任务上花费的时间,CUDA中的事件本质上是一个GPU时间戳,由于CUDA事件是在
	GPU上实现的,因此它们不适于对同时包含设备代码和主机代码的混合代码计时*/
	cudaEvent_t start, stop;
	// cudaEventCreate: 创建一个事件对象,异步启动
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// cudaEventRecord: 记录一个事件,异步启动,start记录起始时间
	cudaEventRecord(start, 0);

	size_t lengthA{ elements_num * sizeof(float) }, lengthB{ elements_num * sizeof(float) };
	float *d_A{ nullptr }, *d_B{ nullptr }, *d_partial_C{ nullptr };

	// cudaMalloc: 在设备端分配内存
	cudaMalloc(&d_A, lengthA);
	cudaMalloc(&d_B, lengthB);

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

	const int threadsPerBlock{ 256 };
	const int blocksPerGrid = std::min(64, (elements_num + threadsPerBlock - 1) / threadsPerBlock);
	size_t lengthC{ blocksPerGrid * sizeof(float) };
	cudaMalloc(&d_partial_C, lengthC);

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
	dot_product << < blocksPerGrid, threadsPerBlock >> >(d_A, d_B, d_partial_C, elements_num);

	/* cudaDeviceSynchronize: kernel的启动是异步的, 为了定位它是否出错, 一
	般需要加上cudaDeviceSynchronize函数进行同步; 将会一直处于阻塞状态,直到
	前面所有请求的任务已经被全部执行完毕,如果前面执行的某个任务失败,将会
	返回一个错误；当程序中有多个流,并且流之间在某一点需要通信时,那就必须
	在这一点处加上同步的语句,即cudaDeviceSynchronize；异步启动
	reference: https://stackoverflow.com/questions/11888772/when-to-call-cudadevicesynchronize */
	//cudaDeviceSynchronize();

	std::unique_ptr<float[]> partial_C(new float[blocksPerGrid]);
	cudaMemcpy(partial_C.get(), d_partial_C, lengthC, cudaMemcpyDeviceToHost);

	*value = 0.f;
	for (int i = 0; i < blocksPerGrid; ++i) {
		(*value) += partial_C[i];
	}

	// cudaFree: 释放设备上由cudaMalloc函数分配的内存
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_partial_C);

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

static int dot_product_gpu_2(const float* A, const float* B, float* value, int elements_num, float* elapsed_time)
{
	// cudaDeviceProp: cuda设备属性结构体
	cudaDeviceProp prop;
	int count;
	// cudaGetDeviceCount: 获得计算能力设备的数量
	cudaGetDeviceCount(&count);
	//fprintf(stderr, "device count: %d\n", count);
	int whichDevice;
	// cudaGetDevice: 获得当前正在使用的设备ID，设备ID从0开始编号
	cudaGetDevice(&whichDevice);
	// cudaGetDeviceProperties: 获取GPU设备相关信息
	cudaGetDeviceProperties(&prop, whichDevice);
	// cudaDeviceProp::canMapHostMemory: GPU是否支持设备映射主机内存
	if (prop.canMapHostMemory != 1) {
		fprintf(stderr, "Device cannot map memory.\n");
		return -1;
	}
	
	// cudaSetDeviceFlags: 设置设备要用于执行的标志
	// 将设备置入能分配零拷贝内存的状态
	cudaSetDeviceFlags(cudaDeviceMapHost);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	const int threadsPerBlock{ 256 };
	const int blocksPerGrid = std::min(64, (elements_num + threadsPerBlock - 1) / threadsPerBlock);

	size_t lengthA{ elements_num * sizeof(float) }, lengthB{ elements_num * sizeof(float) };
	float *d_A{ nullptr }, *d_B{ nullptr }, *d_partial_C{ nullptr };
	float *a{ nullptr }, *b{ nullptr }, *partial_c{ nullptr };

	/* cudaHostAlloc: 分配主机内存。C库函数malloc将分配标准的，可
	分页的(Pagable)主机内存，而cudaHostAlloc将分配页锁定的主机内存。页锁定内
	存也称为固定内存(Pinned Memory)或者不可分页内存，它有一个重要的属性：操作系
	统将不会对这块内存分页并交换到磁盘上，从而确保了该内存始终驻留在物理内
	存中。因此，操作系统能够安全地使某个应用程序访问该内存的物理地址，因为
	这块内存将不会被破坏或者重新定位。由于GPU知道内存的物理地址，因此可以通
	过"直接内存访问(Direct Memory Access, DMA)"技术来在GPU和主机之间复制数据。
	固定内存是一把双刃剑。当使用固定内存时，你将失去虚拟内存的所有功能。
	建议：仅对cudaMemcpy调用中的源内存或者目标内存，才使用页锁定内存，并且在
	不再需要使用它们时立即释放。
	零拷贝内存：通过cudaHostAlloc函数+cudaHostAllocMapped参数，而固定内存是
	cudaHostAlloc函数+cudaHostAllocDefault参数。通过cudaHostAllocMapped分配
	的主机内存也是固定的，它与通过cudaHostAllocDefault分配的固定内存有着相同
	的属性。但这种内存除了可以用于主机与GPU之间的内存复制外，还可以在CUDA C核
	函数中直接访问这种类型的主机内存，而不需要复制到GPU，因此也称为零拷贝内存。
	cudaHostAllocMapped：这个标志告诉运行时将从GPU中访问这块内存。
	cudaHostAllocWriteCombined：这个标志表示，运行时应该将内存分配为"合并式写
	入(Write-Combined)"内存。这个标志并不会改变应用程序的性能，但却可以显著地
	提升GPU读取内存时的性能。然而，当CPU也要读取这块内存时，"合并式写入"会显得
	很低效。
	对于集成GPU，使用零拷贝内存通常都会带来性能提升，因为内存在物理上与主机是
	共享的。将缓冲区声明为零拷贝内存的唯一作用就是避免不必要的数据复制。所有类型
	的固定内存都存在一定的局限性，零拷贝内存同样不例外：每个固定内存都会占用系统
	的可用物理内存，这最终将降低系统的性能。
	当输入内存和输出内存都只能使用一次时，那么在独立GPU上使用零拷贝内存将带来性能提升。 */
	// allocate the memory on the CPU
	cudaHostAlloc(&a, lengthA, cudaHostAllocWriteCombined | cudaHostAllocMapped);
	cudaHostAlloc(&b, lengthB, cudaHostAllocWriteCombined | cudaHostAllocMapped);
	cudaHostAlloc(&partial_c, blocksPerGrid * sizeof(float), cudaHostAllocMapped);

	/* cudaHostGetDevicePointer: 获得由cudaHostAlloc分配的映射主机内存的设备指针。
	由于GPU的虚拟内存空间地址映射与CPU不同，而cudaHostAlloc返回的是CPU上的指针，
	因此需要调用cudaHostGetDevicePointer函数来获得这块内存在GPU上的有效指针。这些指针
	将被传递给核函数，并在随后由GPU对这块内存执行读取和写入等操作 */
	// find out the GPU pointers
	cudaHostGetDevicePointer(&d_A, a, 0);
	cudaHostGetDevicePointer(&d_B, b, 0);
	cudaHostGetDevicePointer(&d_partial_C, partial_c, 0);

	memcpy(a, A, lengthA);
	memcpy(b, B, lengthB);

	dot_product << < blocksPerGrid, threadsPerBlock >> >(d_A, d_B, d_partial_C, elements_num);

	/* cudaThreadSynchronize: 等待计算设备完成, 将CPU与GPU同步*/
	cudaThreadSynchronize();

	*value = 0.f;
	for (int i = 0; i < blocksPerGrid; ++i) {
		(*value) += partial_c[i];
	}

	// cudaFreeHost: 释放设备上由cudaHostAlloc函数分配的内存
	cudaFreeHost(d_A);
	cudaFreeHost(d_B);
	cudaFreeHost(d_partial_C);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(elapsed_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}

int dot_product_gpu(const float* A, const float* B, float* value, int elements_num, float* elapsed_time)
{
	int ret{ 0 };
	//ret = dot_product_gpu_1(A, B, value, elements_num, elapsed_time); // 普通实现
	ret = dot_product_gpu_2(A, B, value, elements_num, elapsed_time); // 通过零拷贝内存实现

	return ret;
}
