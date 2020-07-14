#include "funset.hpp"
#include <omp.h>
#include <thread>
#include <chrono>

// Blog: https://blog.csdn.net/fengbingchun/article/details/15027507

int test_openmp_api()
{
	//omp_set_num_threads(4);
	fprintf(stdout, "get num threads: %d\n", omp_get_num_threads());
	fprintf(stdout, "get max threads: %d\n", omp_get_max_threads());
	fprintf(stdout, "get thread num: %d\n", omp_get_thread_num());

	omp_set_dynamic(4);
	fprintf(stdout, "get num procs: %d\n", omp_get_num_procs());
	fprintf(stdout, "get dynamic: %d\n", omp_get_dynamic());

	fprintf(stdout, "in parallel: %d\n", omp_in_parallel());

	omp_set_nested(2);
	fprintf(stdout, "get nested: %d\n", omp_get_nested());

	// omp_lock_t

	fprintf(stdout, "get wtime: %f\n", omp_get_wtime());
	fprintf(stdout, "get wtick: %f\n", omp_get_wtick());

	return 0;
}

int test_openmp_parallel()
{
	// 注意: 由于多线程执行时序的随机性，每次执行时打印的结果顺序可能存在差别
	// parallel指令是用来产生或唤醒多个线程创建并行域的，并且可以用num_threads子句控制线程数目. parallel域中的每行代码都被
	// 多个线程重复执行parallel的并行域内部代码中，若再出现parallel制导指令则出现并行域嵌套问题，如果设置了OMP_NESTED环境
	// 变量，那么在条件许可时内部并行域也会由多个线程执行，反之没有设置相应变量，那么内部并行域的代码将只有一个线程来执行。
	// 还有一个环境变量OMP_DYNAMIC也影响并行域的行为，如果没有设置该环境变量将不允许动态调整并行域内的线程数目，
	// omp_set_dynamic()也是用于同样的目的
	#pragma omp parallel num_threads(8)
	{
		fprintf(stdout, "hello, world! thread id = %d\n", omp_get_thread_num());
	}

	return 0;
}

int test_openmp_parallel_for()
{
	// 注意: 由于多线程执行时序的随机性，每次执行时打印的结果顺序可能存在差别

{
	// for指令指定紧随它的循环语句必须由线程组并行执行，用来将一个for循环任务分配到多个线程，此时各个线程各自分担其中一部分
	// 工作.for指令要和parallel指令结合起来使用才有效果，即for出现在并行域中才能有多个线程来分担任务
	#pragma omp parallel for
	for (int j = 0; j < 4; ++j) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
		fprintf(stdout, "j = %d, thread id = %d\n", j, omp_get_thread_num());
	}
}
	fprintf(stdout, "\n\n");

{ // for指令一般可以和parallel指令合起来形成parallel for指令使用，也可以单独用在parallel指令的并行域中
	// 此时只有一个并行域，在该并行域内的多个线程首先完成第一个for语句的任务分担，然后在此进行一次同步(for制导指令
	// 本身隐含有结束处的路障同步)，然后再进行第二个for语句的任务分担，直到退出并行域只剩下一个主线程为止
	#pragma omp parallel
	{
		#pragma omp for
		for (int j = 0; j < 8; ++j) {
			std::this_thread::sleep_for(std::chrono::seconds(1));
			fprintf(stdout, "j = %d, thread id = %d\n", j, omp_get_thread_num());
		}

		#pragma omp for
		for (int i = 0; i < 8; ++i) {
			std::this_thread::sleep_for(std::chrono::seconds(1));
			fprintf(stdout, "i = %d, thread id = %d\n", i, omp_get_thread_num());
		}
	}
}

	return 0;
}

int test_openmp_for_schedule()
{
	// 注意: 由于多线程执行时序的随机性，每次执行时打印的结果顺序可能存在差别

	// 当循环中每次迭代的计算量不相等时，如果简单地给各个线程分配相同次数的迭代的话，会使得各个线程计算负载不均衡，这会使得
	// 有些线程先执行完，有些后执行完，造成某些CPU核空闲，影响程序性能. 在OpenMP的for任务分担中，任务的划分称为调度，各个
	// 线程如何划分任务是可以调整的，因此有静态划分、动态划分等，所以调度也分成多个类型。for任务调度子句只能用于for制导
	// 指令中。在OpenMP中，对for循环任务调度使用schedule子句来实现。
	// schedule子句使用格式为：schedule(type[, size]). type参数，表示调度类型，有四种调度类型如下: static、dynamic、guided、
	// runtime。size参数为可选，表示以循环迭代次数计算的划分单位，每个线程所承担的计算任务对应于0个或若干个size次循环，
	// size参数必须是整数。static、dynamic、guided三种调度方式都可以使用size参数，也可以不使用size参数。
	// 当type参数类型为runtime时，size参数是非法的。

{ // static静态调度: 当for或者parallelfor编译制导指令没有带schedule子句时，大部分系统中默认采用size为1的static调度方式
	// 使用size参数时，分配给每个线程的size次连续的迭代计算
	#pragma omp parallel for schedule(static, 2)
	for (int i = 0; i < 10; ++i) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
		fprintf(stdout, "i = %d, thread id = %d\n", i, omp_get_thread_num());
	}
}
	fprintf(stdout, "\n\n");

{ // dynamic动态调整：是动态地将迭代分配到各个线程，各线程动态的申请任务，因此较快的线程可能申请更多次数，而较慢的线程申请
  // 任务次数可能较少，因此动态调整可以在一定程度上避免前面提到的按循环次数划分引起的负载不平衡问题
	// 动态调整时，size小有利于实现更好的负载均衡，但是会引起过多的任务动态申请的开销，反之size大则开销较少，
	// 但是不易于实现负载平衡，size的选择需要在这两者之间进行权衡
	#pragma omp parallel for schedule(dynamic, 2)
	for (int i = 0; i < 10; ++i) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
		fprintf(stdout, "i = %d, thread id = %d\n", i, omp_get_thread_num());
	}
}
	fprintf(stdout, "\n\n");

{ // guided调度：是一种采用指导性的启发式自调度方法。开始时每个线程会分配到较大的迭代块，之后分配到的迭代块会逐渐递减。
  // 迭代块的大小会按指数级下降到指定的size大小，如果没有指定size参数，那么迭代块大小最小会降到1
	#pragma omp parallel for schedule(guided, 2)
	for (int i = 0; i < 10; ++i) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
		fprintf(stdout, "i = %d, thread id = %d\n", i, omp_get_thread_num());
	}
}
	fprintf(stdout, "\n\n");

{ // runtime调度：它不像static、dynamic、guided三种调度方式那样是真实调度方式。它是在运行时根据环境变量OMP_SCHEDULE来确定调度
  // 类型，最终使用的调度类型仍然是static、dynamic、guided中的一种
	#pragma omp parallel for schedule(runtime)
	for (int i = 0; i < 10; ++i) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
		fprintf(stdout, "i = %d, thread id = %d\n", i, omp_get_thread_num());
	}
}

	return 0;
}

int test_openmp_parallel_sections()
{
	// 注意: 由于多线程执行时序的随机性，每次执行时打印的结果顺序可能存在差别

	// sections编译制导指令：是用于非迭代计算的任务分担，它将sections语句里的代码用section制导指令划分成几个不同的
	// 段(可以是一条语句，也可以是用{…}括起来的结构块)，不同的section段由不同的线程并行执行。

{
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			std::this_thread::sleep_for(std::chrono::seconds(1));
			fprintf(stdout, "thread id = %d\n", omp_get_thread_num());
		}
		
		#pragma omp section
		{
			std::this_thread::sleep_for(std::chrono::seconds(1));
			fprintf(stdout, "thread id = %d\n", omp_get_thread_num());
		}

		#pragma omp section
		{
			std::this_thread::sleep_for(std::chrono::seconds(1));
			fprintf(stdout, "thread id = %d\n", omp_get_thread_num());
		}
	}
}
	fprintf(stdout, "\n\n");

{
	// 这里有两个sections构造先后串行执行的，即第二个sections构造的代码要等第一个sections构造的代码执行完后才能执行sections
	// 构造里面的各个section部分代码是并行执行的。与for制导指令一样，在sections的结束处有一个隐含的路障同步，没有其他说明
	// 的情况下，所有线程都必须到达该点才能往下运行。使用section指令时，需要注意的是这种方式需要保证各个section里的代码
	// 执行时间相差不大，否则某个section执行时间比其他section过长就造成了其它线程空闲等待的情况。用for语句来分担任务时工作
	// 量由系统自动划分，只要每次循环间没有时间上的差异，那么分摊是比较均匀的，使用section来划分线程是一种手工划分工作量
	// 的方式，最终负载均衡的好坏得依赖于程序员。
	#pragma omp parallel
	{
		#pragma omp sections
		{
			#pragma omp section
			{
				std::this_thread::sleep_for(std::chrono::seconds(1));
				fprintf(stdout, "xx thread id = %d\n", omp_get_thread_num());
			}
			#pragma omp section
			{
				std::this_thread::sleep_for(std::chrono::seconds(1));
				fprintf(stdout, "xx thread id = %d\n", omp_get_thread_num());
			}
		}

		// 两个sections构造先后串行执行，与for制导指令一样，在sections的结束处有一个隐含的路障同步
		#pragma omp sections
		{
			#pragma omp section
			{
				std::this_thread::sleep_for(std::chrono::seconds(1));
				fprintf(stdout, "yy thread id = %d\n", omp_get_thread_num());
			}
			#pragma omp section
			{
				std::this_thread::sleep_for(std::chrono::seconds(1));
				fprintf(stdout, "yy thread id = %d\n", omp_get_thread_num());
			}
		}
	}
}

	return 0;
}

int test_openmp_parallel_single()
{
	// single制导指令: 单线程执行single制导指令指定所包含的代码只由一个线程执行，别的线程跳过这段代码。
	// 如果没有nowait从句，所有线程在single制导指令结束处隐式同步点同步。如果single制导指令有nowait从句，则别的线程直接向下
	// 执行，不在隐式同步点等待；single制导指令用在一段只被单个线程执行的代码段之前，表示后面的代码段将被单线程执行。

	// 另一种需要使用single制导指令的情况是为了减少并行域创建和撤销的开销，而将多个临界的parallel并行域合并时。经过合并后，原
	// 来并行域之间的串行代码也将被并行执行，违反了代码原来的目的，因此这部分代码可以用single指令加以约束只用一个线程来完成。

	#pragma omp parallel
	{
		#pragma omp single
		{
			std::this_thread::sleep_for(std::chrono::seconds(1));
			fprintf(stdout, "Beginning work1, thread id = %d\n", omp_get_thread_num());
		}

		fprintf(stdout, "work on 1 parallell, thread id = %d\n", omp_get_thread_num());

		#pragma omp single
		{
			std::this_thread::sleep_for(std::chrono::seconds(1));
			fprintf(stdout, "Finishing work1, thread id = %d\n", omp_get_thread_num());
		}

		#pragma omp single nowait
		{
			std::this_thread::sleep_for(std::chrono::seconds(1));
			fprintf(stdout, "Beginning work2, thread id = %d\n", omp_get_thread_num());
		}

		fprintf(stdout, "work on 2 parallelly, thread id = %d\n", omp_get_thread_num());
	}

	return 0;
}

int test_openmp_parallel_critical()
{
	// OpenMP支持两种不同类型的线程同步机制，一种是互斥锁的机制，可以用来保护一块共享的存储空间，
	// 使任何时候访问这块共享内存空间的线程最多只有一个，从而保证了数据的完整性；另外一种同步机制是事件同步机制，
	// 这种机制保证了多个线程之间的执行顺序。

	// critical临界区: 在可能产生内存数据访问竞争的地方，都需要插入相应的临界区制导指令，
	// 格式: #pragam omp critical[(name)]  其中name是可选的，critical语句不允许互相嵌套。

	const int n = 10;
	int arx[n] = { 1, 21, 33, 14, 56, 66, 77, 88, 59, 10 };
	int ary[n] = { 1, 21, 33, 14, 56, 66, 77, 88, 59, 10 };
	int max_num_x = -1;
	int max_num_y = -1;

	// 在一个并行域内的for任务分担域中，各个线程逐个进入到critical保护的区域内，比较当前元素和最大值的关系并可能进行
	// 最大值的更替，从而避免了数据竞争的情况。
	#pragma omp parallel for
	for (int i = 0; i < n; ++i) {
		#pragma omp critical(max_arx)
		if (arx[i] > max_num_x) {
			std::this_thread::sleep_for(std::chrono::milliseconds(500));
			max_num_x = arx[i];
		}

		#pragma omp critical(max_ary)
		if (ary[i]> max_num_y) {
			std::this_thread::sleep_for(std::chrono::milliseconds(500));
			max_num_y = ary[i];
		}
	}

	fprintf(stdout, "max num x: %d, y: %d\n", max_num_x, max_num_y);
	return 0;
}

int test_openmp_parallel_atomic()
{
	// atomic原子操作：critical临界区操作能够作用在任意大小的代码块上，而原子操作只能作用在单条赋值语句中。
	// 能够使用原子语句的前提条件是相应的语句能够转化成一条机器指令，使得相应的功能能够一次执行完毕而不会被打断。
	// C/C++中可用的原子操作："+、-、*、/、&、^、<<、>>"。值得注意的是，当对一个数据进行原子操作保护的时候，就不能对数据
	// 进行临界区的保护，OpenMP运行时并不能在这两种保护机制之间建立配合机制。
	// 用户在针对同一个内存单元使用原子操作的时候，需要在程序所有涉及到该变量并行赋值的部位都加入原子操作的保护。

	int counter = 0;

	#pragma omp parallel
	{
		for (int i = 0; i < 100; ++i) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			// 由于使用atomic语句，则避免了可能出现的数据访问竞争情况，最后的执行结果都是一致的。
			// 而将atomic这一行语句从源程序中删除时，由于有了数据访问的竞争情况，所以最后的执行结果是不确定的。
			#pragma omp atomic
			counter++;
		}
	}

	fprintf(stdout, "counter = %d\n", counter);
	return 0;
}

int test_openmp_parallel_barrier()
{
	// barrier同步路障：路障(barrier)是OpenMP线程的一种同步方法。线程遇到路障时必须等待，直到并行区域内的所有线程都到达了
	// 同一点，才能继续执行下面的代码。在每一个并行域和任务分担域的结束处都会有一个隐含的同步路障，执行此并行域/任务分担域
	// 的线程组在执行完毕本区域代码之前，都需要同步并行域的所有线程。也就是说在parallel、for、sections和single构造的最后，
	// 会有一个隐式的路障。在有些情况下，隐含的同步路障并不能提供有效的同步措施。这时，需要程序员插入明确的同步路障
	// 语句#pragma omp barrier。此时，在并行区域的执行过程中，所有的执行线程都会在同步路障语句上进行同步。

	#pragma omp parallel
	{
		//Initialization();

		// 只有等所有的线程都完成Initialization()初始化操作以后，才能够进行下一步的处理动作，
		// 因此，在此处插入一个明确的同步路障操作以实现线程之间的同步。
		#pragma  omp barrier
		//Process();
	}

	return 0;
}

int test_openmp_parallel_for_nowait()
{
	// nowait：为了避免在循环过程中不必要的同步路障并加快运行速度，可以使用nowait子句除去这个隐式的路障。

	#pragma omp parallel num_threads(4)
	{
		// 线程在完成第一个for循环子任务后，并不需要同步等待，而是直接执行后面的任务，因此出现“-”在“+”前面的情况。
		// nowait子句消除了不必要的同步开销，加快了计算速度.
		#pragma omp for nowait
		for (int i = 0; i < 8; ++i) {
			fprintf(stdout, "+\n");
		}

		#pragma omp for
		for (int j = 0; j < 8; ++j) {
			fprintf(stdout, "-\n");
		}
	}

	return 0;
}

int test_openmp_parallel_for_master()
{
	// master主线程执行：用于指定一段代码由主线程执行。master制导指令和single制导指令类似，区别在于，master制导指令包含的
	// 代码段只由主线程执行，而single制导指令包含的代码段可由任一线程执行，并且master制导指令在结束处没有隐式同步，
	// 也不能指定nowait从句。

	int a[5];

	#pragma omp parallel
	{
		#pragma omp for
		for (int i = 0; i < 5; ++i) {
			a[i] = i * i;
		}

		// 只有一个线程将逐个元素打印出来
		#pragma omp master
		for (int i = 0; i < 5; ++i) {
			fprintf(stdout, "a[%d] = %d\n", i, a[i]);
		}
	}

	return 0;
}

int test_openmp_parallel_for_ordered()
{
	// ordered顺序制导指令：对于循环代码的任务分担中，某些代码的执行需要按规定的顺序执行。典型的情况如下：在一次循环的过程中
	// 大部分的工作是可以并行执行的，而特定部分代码的工作需要等到前面的工作全部完成之后才能够执行。这时，可以使用
	// ordered子句使特定的代码按照串行循环的次序来执行。

	//#pragma omp parallel for ordered
	//for (int i = 0; i < 100; ++i) {
	//	//一些无数据相关、可并行乱序执行的操作
	//	//do someting

	//	//一些有数据相关、只能顺序执行的操作
	//	//do someting
	//	#pragma omp ordered
	//	for (int j = 0; j < 2; ++j)
	//		int tmp = 1;
	//}

	return 0;
}

int test_openmp_lock()
{
	// 互斥锁函数：除了atomic和critical编译制导指令，OpenMP还可以通过库函数支持实现互斥操作，方便用户实现特定的同步需求。
	// 编译制导指令的互斥支持只能放置在一段代码之前，作用在这段代码之上。而OpenMP API所提供的互斥函数可放在任意需要的位置。
	// 程序员必须自己保证在调用相应锁操作之后释放相应的锁，否则就可能造成多线程程序的死锁。
	// 互斥锁函数中只有omp_test_lock函数是带有返回值的，该函数可以看作是omp_set_lock的非阻塞版本。

	omp_lock_t lock;
	omp_init_lock(&lock);

	// 对for循环中的所有内容进行加锁保护，同时只能有一个线程执行for循环中的内容
	#pragma omp parallel for
	for (int i = 0; i < 5; ++i) {
		omp_set_lock(&lock);
		fprintf(stdout, "i = %d, thread id: %d +\n", i, omp_get_thread_num());
		fprintf(stdout, "i = %d, thread id: %d -\n", i, omp_get_thread_num());
		omp_unset_lock(&lock);
	}

	omp_destroy_lock(&lock);
	return 0;
}

int test_openmp_variable_private()
{
	// OpenMP各个线程的变量是共享还是私有，是依据OpenMP自身的规则和相关的数据子句而定，而不是依据操作系统线程或进程上的变量
	// 特性而定。OpenMP的数据处理子句包括private、firstprivate、lastprivate、shared、default、reduction copyin和copyprivate.
	// 它与编译制导指令parallel、for和sections相结合用来控制变量的作用范围。它们控制数据变量，比如，哪些串行部分中的数据变量
	// 被传递到程序的并行部分以及如何传送，哪些变量对所有并行部分的线程是可见的，哪些变量对所有并行部分的线程是私有的，等等。

	// shared子句：用来声明一个或多个变量是共享变量。需要注意的是，在并行域内使用共享变量时，如果存在写操作，必须对共享变量
	// 加以保护，否则不要轻易使用共享变量，尽量将共享变量的访问转化为私有变量的访问。
	// 循环迭代变量在循环构造的任务分担域里是私有的。声明在任务分担域内的自动变量都是私有的。

	// default子句：用来允许用户控制并行区域中变量的共享属性。使用shared时，缺省情况下，传入并行区域内的同名变量被当作共享
	// 变量来处理，不会产生线程私有副本，除非使用private等子句来指定某些变量为私有的才会产生副本。
	// 如果使用none作为参数，除了那些由明确定义的除外，线程中用到的变量都必须显式指定为是共享的还是私有的。

	// private子句：用来将一个或多个变量声明成线程私有的变量，变量声明成私有变量后，指定每个线程都有它自己的变量私有副本，
	// 其他线程无法访问私有副本。即使在并行域外有同名的共享变量，共享变量在并行域内不起任何作用，并且并行域内不会操作到
	// 外面的共享变量。出现在reduction子句中的变量不能出现在private子句中。

	int k = 100;

	// for循环前的变量k和循环区域内的变量k其实是两个不同的变量。
	// 用private子句声明的私有变量的初始值在并行域的入口处是未定义的，它并不会继承同名共享变量的值。
	#pragma omp parallel for private(k)
	for (k = 0; k < 8; ++k) {
		std::this_thread::sleep_for(std::chrono::milliseconds(500));
		fprintf(stdout, "k = %d, thread id: %d\n", k, omp_get_thread_num());
	}

	fprintf(stdout, "last k = %d\n", k); // 100
	return 0;
}

int test_openmp_variable_firstprivate()
{
	// firstprivate子句：私有变量的初始化和终结操作，OpenMP编译制导指令需要对这种需求给予支持，即使用firstprivate和
	// lastprivate来满足这两种需求。使得并行域或任务分担域开始执行时，私有变量通过主线程中的变量初始化，也可以在并行域或任务
	// 分担结束时，将最后一次一个线程上的私有变量赋值给主线程的同名变量。private声明的私有变量不会继承同名变量的值，于是
	// OpenMP提供了firstprivate子句来实现这个功能。
	// firstprivate子句是private子句的超集，即不仅包含了private子句的功能，而且还要对变量进行初始化。

	int i = 5, k = 100;

	// 并行域内的私有变量k继承了外面共享变量k的值100作为初始值，并且在退出并行区域后，共享变量k的值保持为100未变
	#pragma omp parallel for firstprivate(k)
	for (i = 0; i < 4; ++i) {
		std::this_thread::sleep_for(std::chrono::milliseconds(500));
		k += i;
		fprintf(stdout, "k = %d, thread id: %d\n", k, omp_get_thread_num());
	}

	fprintf(stdout, "last k = %d\n", k); // 100
	return 0;
}

int test_openmp_variable_lastprivate()
{
	// lastprivate子句：有时要将任务分担域内私有变量的值经过计算后，在退出时，将它的值赋给同名的共享变量(private和
	// firstprivate子句在退出并行域时都没有将私有变量的最后取值赋给对应的共享变量)，lastprivate子句就是用来实现在退出并行
	// 域时将私有变量的值赋给共享变量。lastprivate子句也是private子句的超集，即不仅包含了private子句的功能，而且还要将变量
	// 从for、sections的任务分担域中最后的线程中复制给外部同名变量。由于在并行域内是多个线程并行执行的，最后到底是将哪个线
	// 程的最终计算结果赋给了对应的共享变量呢？OpenMP规范中指出，如果是for循环迭代，那么是将最后一次循环迭代中的值赋给对应
	// 的共享变量；如果是sections构造，那么是代码中排在最后的section语句中的值赋给对应的共享变量。注意这里说的最后一个
	// section是指程序语法上的最后一个，而不是实际运行时的最后一个运行完的。

	int i = 5, k = 100;

	// 退出for循环的并行区域后，共享变量k的值变成了103，而不是保持原来的100不变
	#pragma omp parallel for firstprivate(k), lastprivate(k)
	for (i = 0; i < 4; ++i) {
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		k += i;
		fprintf(stdout, "k = %d, thread id: %d\n", k, omp_get_thread_num());
	}

	fprintf(stdout, "last k = %d\n", k); // 103
	return 0;
}

int test_openmp_variable_flush()
{
	// flush：OpenMP的flush制导指令主要与多个线程之间的共享变量的一致性问题。用法：flush[(list)]，
	// 该指令将列表中的变量执行flush操作，直到所有变量都已完成相关操作后才返回，保证了后续变量访问的一致性。

	return 0;
}

namespace {
int counter = 0;

// 实现一个线程私有的计数器，各个线程使用同一个函数来实现自己的计数
#pragma omp threadprivate(counter)
inline int increment_counter() { return ++counter; }
} // namespace

int test_openmp_variable_threadprivate()
{
	// 线程专有数据：它和私有数据不太相同，threadprivate子句用来指定全局的对象被各个线程各自复制了一个私有的拷贝，即各个线
	// 程具有各自私有、线程范围内的全局对象。private变量在退出并行域后则失效，而threadprivate线程专有变量可以在前后
	// 多个并行域之间保持连续性。

	// threadprivate子句：用作threadprivate的变量的地址不能是常数。对于C++的类(class)类型变量，用作threadprivate的参数时
	// 有些限制，当定义时带有外部初始化则必须具有明确的拷贝构造函数。
	// 对于windows系统，threadprivate不能用于动态装载(使用LoadLibrary装载)的DLL中，可以用于静态装载的DLL中。

	#pragma omp parallel
	{
		#pragma omp for
		for (int j = 0; j < 8; ++j) {
			std::this_thread::sleep_for(std::chrono::seconds(1));
			fprintf(stdout, "counter = %d, thread id = %d\n", increment_counter(), omp_get_thread_num());
		}

		#pragma omp for
		for (int i = 0; i < 8; ++i) {
			std::this_thread::sleep_for(std::chrono::seconds(1));
			fprintf(stdout, "counter = %d, thread id = %d\n", increment_counter(), omp_get_thread_num());
		}
	}

	return 0;
}

namespace {
int counter2 = 0;

// threadprivate中的计数器函数，如果多个线程使用时，各个线程都需要对全局变量counter2的副本进行初始化。
#pragma omp threadprivate(counter2)
inline int increment_counter2() { return ++counter2; }
} // namespace

int test_openmp_variable_copyin()
{
	// copyin子句：用来将主线程中threadprivate变量的值复制到执行并行域的各个线程的threadprivate变量中，便于所有线程访问
	// 主线程中的变量值。copyin中的参数必须被声明成threadprivate的，对于class类型的变量，必须带有明确的拷贝赋值操作符。

	#pragma omp parallel sections copyin(counter2)
	{
		#pragma omp section
		{
			int count1;
			for (int iterator = 0; iterator < 100; ++iterator) {
				count1 = increment_counter2();
			}
			fprintf(stdout, "count1 = %d\n", count1); // 100
		}

		#pragma omp section
		{
			int count2;
			for (int iterator = 0; iterator < 200; ++iterator) {
				count2 = increment_counter2();
			}
			fprintf(stdout, "count2 = %d\n", count2); // 200
		}
	}

	fprintf(stdout, "counter2 = %d\n", counter2); // 0

	#pragma omp parallel sections copyin(counter2)
	{
		#pragma omp section
		{
			int count1;
			for (int iterator = 0; iterator < 100; ++iterator) {
				count1 = increment_counter2();
			}
			fprintf(stdout, "count1 = %d\n", count1); // 100
		}

		#pragma omp section
		{
			int count2;
			for (int iterator = 0; iterator < 200; ++iterator) {
				count2 = increment_counter2();
			}
			fprintf(stdout, "count2 = %d\n", count2); // 200
		}
	}

	fprintf(stdout, "counter2 = %d\n", counter2); // 0
	return 0;
}

namespace {
int counter3 = 0;

#pragma omp threadprivate(counter3)
inline int increment_counter3() { return ++counter3; }
} // namespace

int test_openmp_variable_copyprivate()
{
	// copyprivate子句：提供了一种机制，即将一个线程私有变量的值广播到执行同一并行域的其他线程。copyprivate子句可以关联
	// single构造，在single构造的barrier到达之前就完成了广播工作。copyprivate可以对private和threadprivate子句中的变量进
	// 行操作，但是当使用single构造时，copyprivate的变量不能用于private和firstprivate子句中。

	#pragma omp parallel
	{
		int count;
		// 使用copyprivate子句后，single构造内给counter3赋的值被广播到了其它线程里，但没有使用copyprivate子句时，
		// 只有一个线程获得了single构造内的赋值，其它线程没有获取single构造内的赋值。
		#pragma omp single copyprivate(counter3)
		{
			counter3 = 50;
		}

		count = increment_counter3();
		fprintf(stdout, "thread id = %d, count = %d\n", omp_get_thread_num(), count); // count = 51
	}

	return 0;
}

int test_openmp_variable_reduction()
{
	// 归约操作：reduction子句主要用来对一个或多个参数条目指定一个操作符，每个线程将创建参数条目的一个私有拷贝，在并行域
	// 或任务分担域的结束处，将用私有拷贝的值通过指定的运行符运算，原始的参数条目被运算结果的值更新。列出了可以用于
	// reduction子句的一些操作符以及对应私有拷贝变量缺省的初始值，私有拷贝变量的实际初始值依赖于reduction变量的数据类
	// 型：+(0)、-(0)、*(1)、&(~0)、|(0)、^(0)、&&(1)、||(0)。如果在并行域内不加锁保护就直接对共享变量进行写操作，存在数据
	// 竞争问题，会导致不可预测的异常结果。如果共享数据作为private、firstprivate、lastprivate、threadprivate、reduction子
	// 句的参数进入并行域后，就变成线程私有了，不需要加锁保护了。

	int sum = 2;

	#pragma omp parallel for reduction(+:sum)
	for (int i = 0; i < 5; ++i) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
		fprintf(stdout, "thread id = %d, i = %d, sum = %d\n", omp_get_thread_num(), i, sum);
		sum += i;

	}

	fprintf(stdout, "sum = %d\n", sum); // sum = 12
	return 0;
}
