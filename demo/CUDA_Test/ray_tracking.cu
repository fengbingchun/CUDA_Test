#include "funset.hpp"
#include <iostream>
#include <algorithm>
#include <memory>
#include <vector>
#include <cuda_runtime.h> // For the CUDA runtime routines (prefixed with "cuda_")
#include <device_launch_parameters.h>
#include "common.hpp"

// 通过一个数据结构对球面建模
struct Sphere {
	float r, b, g;
	float radius;
	float x, y, z;
	/* __device__: 函数类型限定符,表明被修饰的函数在设备上执行，只能从设备上调用，
	但只能在其它__device__函数或者__global__函数中调用；__device__函数不支持递归；
	__device__函数的函数体内不能声明静态变量；__device__函数的参数数目是不可变化的;
	不能对__device__函数取指针 */
	__device__ float hit(float ox, float oy, float *n)
	{
		float dx = ox - x;
		float dy = oy - y;
		if (dx*dx + dy*dy < radius*radius) {
			float dz = sqrtf(radius*radius - dx*dx - dy*dy);
			*n = dz / sqrtf(radius * radius);
			return dz + z;
		}
		return -INF;
	}
};

// method2: 使用常量内存
/* __constant__: 变量类型限定符，或者与__device__限定符连用，这样声明的变量：存
在于常数存储器空间；与应用程序具有相同的生命周期；可以通过运行时库从主机端访问，
设备端的所有线程也可访问。__constant__变量默认为是静态存储。__constant__不能用
extern关键字声明为外部变量。__constant__变量只能在文件作用域中声明，不能再函数
体内声明。__constant__变量不能从device中赋值，只能从host中通过host运行时函数赋
值。__constant__将把变量的访问限制为只读。与从全局内存中读取数据相比，从常量内
存中读取相同的数据可以节约内存带宽。常量内存用于保存在核函数执行期间不会发生变
化的数据。
常量内存：用于保存在核函数执行期间不会发生变化的数据。NVIDIA硬件提供了64KB的常
量内存，并且对常量内存采取了不同于标准全局内存的处理方式。在某些情况中，用常量
内存来替换全局内存能有效地减少内存带宽。 在某些情况下，使用常量内存将提升应用程
序的性能 */
__constant__ Sphere dev_spheres[20]; // 常量内存, = sphere_num

/* __global__: 函数类型限定符;在设备上运行;在主机端调用,计算能力3.2及以上可以在
设备端调用;声明的函数的返回值必须是void类型;对此类型函数的调用是异步的,即在
设备完全完成它的运行之前就返回了;对此类型函数的调用必须指定执行配置,即用于在
设备上执行函数时的grid和block的维度,以及相关的流(即插入<<<   >>>运算符);
a kernel,表示此函数为内核函数(运行在GPU上的CUDA并行计算函数称为kernel(内核函
数),内核函数必须通过__global__函数类型限定符定义); */
__global__ static void ray_tracking(unsigned char* ptr_image, Sphere* ptr_sphere, int width, int height, int sphere_num)
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
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	float ox{ (x - width / 2.f) };
	float oy{ (y - height / 2.f) };

	float r{ 0 }, g{ 0 }, b{ 0 };
	float maxz{ -INF };

	for (int i = 0; i < sphere_num; ++i) {
		float n;
		float t = ptr_sphere[i].hit(ox, oy, &n);
		if (t > maxz) {
			float fscale = n;
			r = ptr_sphere[i].r * fscale;
			g = ptr_sphere[i].g * fscale;
			b = ptr_sphere[i].b * fscale;
			maxz = t;
		}
	}

	ptr_image[offset * 4 + 0] = static_cast<unsigned char>(r * 255);
	ptr_image[offset * 4 + 1] = static_cast<unsigned char>(g * 255);
	ptr_image[offset * 4 + 2] = static_cast<unsigned char>(b * 255);
	ptr_image[offset * 4 + 3] = 255;
}

__global__ static void ray_tracking(unsigned char* ptr_image, int width, int height, int sphere_num)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	float ox{ (x - width / 2.f) };
	float oy{ (y - height / 2.f) };

	float r{ 0 }, g{ 0 }, b{ 0 };
	float maxz{ -INF };

	for (int i = 0; i < sphere_num; ++i) {
		float n;
		float t = dev_spheres[i].hit(ox, oy, &n);
		if (t > maxz) {
			float fscale = n;
			r = dev_spheres[i].r * fscale;
			g = dev_spheres[i].g * fscale;
			b = dev_spheres[i].b * fscale;
			maxz = t;
		}
	}

	ptr_image[offset * 4 + 0] = static_cast<unsigned char>(r * 255);
	ptr_image[offset * 4 + 1] = static_cast<unsigned char>(g * 255);
	ptr_image[offset * 4 + 2] = static_cast<unsigned char>(b * 255);
	ptr_image[offset * 4 + 3] = 255;
}

int ray_tracking_gpu(const float* a, const float* b, const float* c, int sphere_num, unsigned char* ptr, int width, int height, float* elapsed_time)
{
	/* cudaEvent_t: CUDA event types,结构体类型, CUDA事件,用于测量GPU在某
	个任务上花费的时间,CUDA中的事件本质上是一个GPU时间戳,由于CUDA事件是在
	GPU上实现的,因此它们不适于对同时包含设备代码和主机代码的混合代码计时 */
	cudaEvent_t start, stop;
	// cudaEventCreate: 创建一个事件对象,异步启动
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// cudaEventRecord: 记录一个事件,异步启动,start记录起始时间
	cudaEventRecord(start, 0);

	const size_t length{ width * height * 4 * sizeof(unsigned char) };
	unsigned char* dev_image{ nullptr };

	std::unique_ptr<Sphere[]> spheres(new Sphere[sphere_num]);
	for (int i = 0, t = 0; i < sphere_num; ++i, t += 3) {
		spheres[i].r = a[t];
		spheres[i].g = a[t + 1];
		spheres[i].b = a[t + 2];
		spheres[i].x = b[t];
		spheres[i].y = b[t + 1];
		spheres[i].z = b[t + 2];
		spheres[i].radius = c[i];
	}

	// cudaMalloc: 在设备端分配内存
	cudaMalloc(&dev_image, length);

	// method1: 没有使用常量内存
	//Sphere* dev_spheres{ nullptr };
	//cudaMalloc(&dev_spheres, sizeof(Sphere) * sphere_num);
	/* cudaMemcpy: 在主机端和设备端拷贝数据,此函数第四个参数仅能是下面之一:
	(1). cudaMemcpyHostToHost: 拷贝数据从主机端到主机端
	(2). cudaMemcpyHostToDevice: 拷贝数据从主机端到设备端
	(3). cudaMemcpyDeviceToHost: 拷贝数据从设备端到主机端
	(4). cudaMemcpyDeviceToDevice: 拷贝数据从设备端到设备端
	(5). cudaMemcpyDefault: 从指针值自动推断拷贝数据方向,需要支持
	统一虚拟寻址(CUDA6.0及以上版本)
	cudaMemcpy函数对于主机是同步的 */
	//cudaMemcpy(dev_spheres, spheres.get(), sizeof(Sphere) * sphere_num, cudaMemcpyHostToDevice);

	// method2: 使用常量内存
	/* cudaMemcpyToSymbol: cudaMemcpyToSymbol和cudaMemcpy参数为
	cudaMemcpyHostToDevice时的唯一差异在于cudaMemcpyToSymbol会复制到常量内
	存，而cudaMemcpy会复制到全局内存 */
	cudaMemcpyToSymbol(dev_spheres, spheres.get(), sizeof(Sphere)* sphere_num);

	const int threads_block{ 16 };
	/* dim3: 基于uint3定义的内置矢量类型，相当于由3个unsigned int类型组成的
	结构体，可表示一个三维数组，在定义dim3类型变量时，凡是没有赋值的元素都
	会被赋予默认值1 */
	dim3 blocks(width / threads_block, height / threads_block);
	dim3 threads(threads_block, threads_block);

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
	//ray_tracking << <blocks, threads >> >(dev_image, dev_spheres, width, height, sphere_num); // method1, 不使用常量内存
	ray_tracking << <blocks, threads >> >(dev_image, width, height, sphere_num); // method2, 使用常量内存

	cudaMemcpy(ptr, dev_image, length, cudaMemcpyDeviceToHost);

	// cudaFree: 释放设备上由cudaMalloc函数分配的内存
	cudaFree(dev_image);
	//cudaFree(dev_spheres); // 使用method1时需要释放, 如果使用常量内存即method2则不需要释放

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
