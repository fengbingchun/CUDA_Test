#include "funset.hpp"
#include <math.h>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "common.hpp"

int test_quaternion()
{
	// Blog: https://blog.csdn.net/fengbingchun/article/details/130790531
	Eigen::Matrix3d matrix3d;
	matrix3d << 2.044289726145588e-004, -0.2587487517264626, -0.9659446369688031,
		-0.9993063898830017, -3.602307923217642e-002, 9.438056030485108e-003,
		-3.723838540803551e-002, 0.9652727185840433, -0.2585766451355823;

	// 1. rotate matrix to quaternion(Hamilton)
	Eigen::Quaterniond q;
	q = matrix3d;
	std::cout.precision(16);
	std::cout << "w:" << q.w() << ", x:" << q.x() << ", y:" << q.y() << ", z:" << q.z() << "\n";

	// 2. quaternion(Hamilton) to rotate matrix
	Eigen::Quaterniond q2(q.w(), q.x(), q.y(), q.z());
	Eigen::Matrix3d R = q2.toRotationMatrix();
	std::cout << "R:\n" << R << "\n";

	return 0;
}

int test_calcCovarMatrix()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/73558370
	// reference: https://stackoverflow.com/questions/15138634/eigen-is-there-an-inbuilt-way-to-calculate-sample-covariance
	std::vector<std::vector<float>> vec{ { 1.2f, 2.5f, 5.6f, -2.5f },
					{ -3.6f, 9.2f, 0.5f, 7.2f },
					{ 4.3f, 1.3f, 9.4f, -3.4f } };
	const int rows{ 3 }, cols{ 4 };

	std::vector<float> vec_;
	for (int i = 0; i < rows; ++i) {
		vec_.insert(vec_.begin() + i * cols, vec[i].begin(), vec[i].end());
	}
	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m(vec_.data(), rows, cols);

	fprintf(stderr, "source matrix:\n");
	std::cout << m << std::endl;

	fprintf(stdout, "\nEigen implement:\n");
	const int nsamples = rows;
	float scale = 1. / (nsamples /*- 1*/);

	Eigen::MatrixXf mean = m.colwise().mean();
	std::cout << "print mean: " << std::endl << mean << std::endl;

	Eigen::MatrixXf tmp(rows, cols);
	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < cols; ++x) {
			tmp(y, x) = m(y, x) - mean(0, x);
		}
	}
	//std::cout << "tmp: " << std::endl << tmp << std::endl;

	Eigen::MatrixXf covar = (tmp.adjoint() * tmp) /*/ float(nsamples - 1)*/;
	std::cout << "print covariance matrix: " << std::endl << covar << std::endl;

	return 0;
}

int test_meanStdDev()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/73323475
	std::vector<std::vector<float>> vec{ { 1.2f, 2.5f, 5.6f, -2.5f },
						{ -3.6f, 9.2f, 0.5f, 7.2f },
						{ 4.3f, 1.3f, 9.4f, -3.4f } };
	const int rows{ 3 }, cols{ 4 };

	std::vector<float> vec_;
	for (int i = 0; i < rows; ++i) {
		vec_.insert(vec_.begin() + i * cols, vec[i].begin(), vec[i].end());
	}
	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m(vec_.data(), 1, rows * cols);

	fprintf(stderr, "source matrix:\n");
	std::cout << m << std::endl;

	Eigen::MatrixXf mean = m.rowwise().mean(); //<==> m.rowwise().sum() / m.cols();
	float mean_ = mean(0, 0);
	Eigen::MatrixXf sqsum = (m * m.transpose()).rowwise().sum();
	float sqsum_ = sqsum(0, 0);
	float scale = 1. / (rows*cols);
	float variance_ = sqsum_ * scale - mean_ * mean_;
	float stddev_ = std::sqrt(variance_);
	fprintf(stdout, "\nEigen implement:\n");
	fprintf(stdout, "mean: %f, variance: %f, standard deviation: %f\n", mean_, variance_, stddev_);

	return 0;
}

int test_trace()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/72895976
	std::vector<std::vector<float>> vec{ { 1.2f, 2.5f, 5.6f, -2.5f },
					{ -3.6f, 9.2f, 0.5f, 7.2f },
					{ 4.3f, 1.3f, 9.4f, -3.4f } };
	const int rows{ 3 }, cols{ 4 };

	std::vector<float> vec_;
	for (int i = 0; i < rows; ++i) {
		vec_.insert(vec_.begin() + i * cols, vec[i].begin(), vec[i].end());
	}
	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m(vec_.data(), rows, cols);

	fprintf(stderr, "source matrix:\n");
	std::cout << m << std::endl;

	float tr = m.trace();
	fprintf(stderr, "\nEigen implement trace: %f\n", tr);

	return 0;
}

int test_pseudoinverse()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/72874623
	//std::vector<std::vector<float>> vec{ { 0.68f, 0.597f },
	//				{ -0.211f, 0.823f },
	//				{ 0.566f, -0.605f } };
	//const int rows{ 3 }, cols{ 2 };

	std::vector<std::vector<float>> vec{ { 0.68f, 0.597f, -0.211f },
					{ 0.823f, 0.566f, -0.605f } };
	const int rows{ 2 }, cols{ 3 };

	std::vector<float> vec_;
	for (int i = 0; i < rows; ++i) {
		vec_.insert(vec_.begin() + i * cols, vec[i].begin(), vec[i].end());
	}
	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m(vec_.data(), rows, cols);

	fprintf(stderr, "source matrix:\n");
	std::cout << m << std::endl;

	fprintf(stderr, "\nEigen implement pseudoinverse:\n");
	auto svd = m.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);

	const auto &singularValues = svd.singularValues();
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> singularValuesInv(m.cols(), m.rows());
	singularValuesInv.setZero();
	double  pinvtoler = 1.e-6; // choose your tolerance wisely
	for (unsigned int i = 0; i < singularValues.size(); ++i) {
		if (singularValues(i) > pinvtoler)
			singularValuesInv(i, i) = 1.0f / singularValues(i);
		else
			singularValuesInv(i, i) = 0.f;
	}

	Eigen::MatrixXf pinvmat = svd.matrixV() * singularValuesInv * svd.matrixU().transpose();
	std::cout << pinvmat << std::endl;

	return 0;
}

int test_SVD()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/72853757
	//std::vector<std::vector<float>> vec{ { 1.2f, 2.5f, 5.6f, -2.5f },
	//				{ -3.6f, 9.2f, 0.5f, 7.2f },
	//				{ 4.3f, 1.3f, 9.4f, -3.4f },
	//				{ 6.4f, 0.1f, -3.7f, 0.9f } };
	//const int rows{ 4 }, cols{ 4 };

	//std::vector<std::vector<float>> vec{ { 1.2f, 2.5f, 5.6f, -2.5f },
	//				{ -3.6f, 9.2f, 0.5f, 7.2f },
	//				{ 4.3f, 1.3f, 9.4f, -3.4f } };
	//const int rows{ 3 }, cols{ 4 };

	std::vector<std::vector<float>> vec{ { 0.68f, 0.597f },
					{ -0.211f, 0.823f },
					{ 0.566f, -0.605f } };
	const int rows{ 3 }, cols{ 2 };

	std::vector<float> vec_;
	for (int i = 0; i < rows; ++i) {
		vec_.insert(vec_.begin() + i * cols, vec[i].begin(), vec[i].end());
	}
	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m(vec_.data(), rows, cols);

	fprintf(stderr, "source matrix:\n");
	std::cout << m << std::endl;

	Eigen::JacobiSVD<Eigen::MatrixXf> svd(m, Eigen::ComputeFullV | Eigen::ComputeFullU); // ComputeThinU | ComputeThinV
	Eigen::MatrixXf singular_values = svd.singularValues();
	Eigen::MatrixXf left_singular_vectors = svd.matrixU();
	Eigen::MatrixXf right_singular_vectors = svd.matrixV();

	fprintf(stderr, "singular values:\n");
	print_matrix(singular_values.data(), singular_values.rows(), singular_values.cols());
	fprintf(stderr, "left singular vectors:\n");
	print_matrix(left_singular_vectors.data(), left_singular_vectors.rows(), left_singular_vectors.cols());
	fprintf(stderr, "right singular vecotrs:\n");
	print_matrix(right_singular_vectors.data(), right_singular_vectors.rows(), right_singular_vectors.cols());

	return 0;
}

int test_eigenvalues_eigenvectors()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/72801310
	std::vector<float> vec{ 1.23f, 2.12f, -4.2f,
		2.12f, -5.6f, 8.79f,
		-4.2f, 8.79f, 7.3f };
	const int N{ 3 };

	fprintf(stderr, "source matrix:\n");
	print_matrix(vec.data(), N, N);
	fprintf(stderr, "\n");

	Eigen::Map<Eigen::MatrixXf> m(vec.data(), N, N);

	Eigen::EigenSolver<Eigen::MatrixXf> es(m);
	Eigen::VectorXf eigen_values = es.eigenvalues().real();
	fprintf(stderr, "eigen values:\n");
	print_matrix(eigen_values.data(), N, 1);
	Eigen::MatrixXf eigen_vectors = es.eigenvectors().real();
	fprintf(stderr, "eigen vectors:\n");
	print_matrix(eigen_vectors.data(), N, N);

	return 0;
}

int test_norm()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/72636374
	fprintf(stderr, "test norm with eigen:\n");
	// 1. vector
	std::vector<float> vec1{ -2, 3, 1 };
	Eigen::VectorXf v(vec1.size());
	for (int i = 0; i < vec1.size(); ++i) {
		v[i] = vec1[i];
	}

	double value = v.lpNorm<Eigen::Infinity>();
	fprintf(stderr, "vector: Inf: %f\n", value);
	value = v.lpNorm<1>();
	fprintf(stderr, "vector: L1: %f\n", value);
	value = v.norm(); // <==> sqrt(v.squaredNorm()) <==> v.lpNorm<2>()
	fprintf(stderr, "vector: L2: %f\n", value);

	// 2. matrix
	std::vector<float> vec2{ -3, 2, 0, 5, 6, 2, 7, 4, 8 };
	const int row_col{ 3 };
	Eigen::Map<Eigen::MatrixXf> m(vec2.data(), row_col, row_col);

	value = m.lpNorm<Eigen::Infinity>();
	fprintf(stderr, "matrix: Inf: %f\n", value);
	value = m.lpNorm<1>();
	fprintf(stderr, "matrix: L1: %f\n", value);
	value = m.norm();
	fprintf(stderr, "matrix: L2: %f\n", value);

	return 0;
}

int test_inverse_matrix()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/72571800
	std::vector<float> vec{ 5, -2, 2, 7, 1, 0, 0, 3, -3, 1, 5, 0, 3, -1, -9, 4 };
	const int N{ 4 };
	if (vec.size() != (int)pow(N, 2)) {
		fprintf(stderr, "vec must be N^2\n");
		return -1;
	}

	Eigen::Map<Eigen::MatrixXf> map(vec.data(), 4, 4);
	Eigen::MatrixXf inv = map.inverse();

	fprintf(stderr, "source matrix:\n");
	print_matrix<float>(vec.data(), N, N);
	fprintf(stderr, "eigen inverse matrix:\n");
	print_matrix<float>(inv.data(), N, N);

	return 0;
}

int test_mat_determinant()
{
	std::vector<float> vec{ 1, 0, 2, -1, 3, 0, 0, 5, 2, 1, 4, -3, 1, 0, 5, 0 };

	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> map(vec.data(), 4, 4);
	double det = map.determinant();
	fprintf(stderr, "det: %f\n", det);

	return 0;
}

int test_mat_transpose()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/71514010
#ifdef __linux__
	const std::vector<std::string> image_name{"test_data/test1.jpg", "test_data/ret_mat_transpose.jpg"};
#else
	const std::vector<std::string> image_name{ "../../../test_data/test1.jpg",
		"../../../test_data/ret_mat_transpose.jpg" };
#endif
	cv::Mat mat_src = cv::imread(image_name[0]);
	if (!mat_src.data) {
		fprintf(stderr, "read image fail: %s\n", image_name[0].c_str());
		return -1;
	}

	std::vector<cv::Mat> mat_split;
	cv::split(mat_src, mat_split);

	// reference: http://stackoverflow.com/questions/14783329/opencv-cvmat-and-eigenmatrix
	// Map the OpenCV matrix with Eigen:
	Eigen::Map<Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
		map_b(mat_split[0].data, mat_split[0].rows, mat_split[0].cols),
		map_g(mat_split[1].data, mat_split[1].rows, mat_split[1].cols),
		map_r(mat_split[2].data, mat_split[2].rows, mat_split[2].cols);

	Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
		trans_b = map_b.transpose(),
		trans_g = map_g.transpose(),
		trans_r = map_r.transpose();

	cv::Mat
		mat_dst_b(trans_b.rows(), trans_b.cols(), CV_8UC1, trans_b.data()),
		mat_dst_g(trans_g.rows(), trans_g.cols(), CV_8UC1, trans_g.data()),
		mat_dst_r(trans_r.rows(), trans_r.cols(), CV_8UC1, trans_r.data());
	std::vector<cv::Mat> mat_merge;
	mat_merge.push_back(std::move(mat_dst_b));
	mat_merge.push_back(std::move(mat_dst_g));
	mat_merge.push_back(std::move(mat_dst_r));
	cv::Mat mat_dst;
	cv::merge(mat_merge, mat_dst);

	cv::imwrite(image_name[1], mat_dst);

	return 0;
}

template <typename T>
static void matrix_mul_matrix(T* p1, int iRow1, int iCol1, T* p2, int iRow2, int iCol2, T* p3)
{
	if (iRow1 != iRow2) return;

	//列优先, Matrix创建的矩阵默认是按列存储
	//Eigen::Map< Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > map1(p1, iRow1, iCol1);
	//Eigen::Map< Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > map2(p2, iRow2, iCol2);
	//Eigen::Map< Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > map3(p3, iCol1, iCol2);

	//行优先
	Eigen::Map< Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > map1(p1, iRow1, iCol1);
	Eigen::Map< Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > map2(p2, iRow2, iCol2);
	Eigen::Map< Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > map3(p3, iCol1, iCol2);

	map3 = map1 * map2;
}

int test_eigen_base()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/47378515
	//1. 矩阵的定义
	Eigen::MatrixXd m(2, 2);
	Eigen::Vector3d vec3d;
	Eigen::Vector4d vec4d(1.0, 2.0, 3.0, 4.0);

	//2. 动态矩阵、静态矩阵
	Eigen::MatrixXd matrixXd;
	Eigen::Matrix3d matrix3d;

	//3. 矩阵元素的访问
	m(0, 0) = 1;
	m(0, 1) = 2;
	m(1, 0) = m(0, 0) + 3;
	m(1, 1) = m(0, 0) * m(0, 1);
	std::cout << m << std::endl << std::endl;

	//4. 设置矩阵的元素
	m << -1.5, 2.4,
		6.7, 2.0;
	std::cout << m << std::endl << std::endl;
	int row = 4;
	int col = 5;
	Eigen::MatrixXf matrixXf(row, col);
	matrixXf << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20;
	std::cout << matrixXf << std::endl << std::endl;
	matrixXf << Eigen::MatrixXf::Identity(row, col);
	std::cout << matrixXf << std::endl << std::endl;

	//5. 重置矩阵大小
	Eigen::MatrixXd matrixXd1(3, 3);
	m = matrixXd1;
	std::cout << m.rows() << "  " << m.cols() << std::endl << std::endl;

	//6. 矩阵运算
	m << 1, 2, 7,
		3, 4, 8,
		5, 6, 9;
	std::cout << m << std::endl;
	matrixXd1 = Eigen::Matrix3d::Random();
	m += matrixXd1;
	std::cout << m << std::endl << std::endl;
	m *= 2;
	std::cout << m << std::endl << std::endl;
	std::cout << -m << std::endl << std::endl;
	std::cout << m << std::endl << std::endl;

	//7. 求矩阵的转置、共轭矩阵、伴随矩阵
	std::cout << m.transpose() << std::endl << std::endl;
	std::cout << m.conjugate() << std::endl << std::endl;
	std::cout << m.adjoint() << std::endl << std::endl;
	std::cout << m << std::endl << std::endl;
	m.transposeInPlace();
	std::cout << m << std::endl << std::endl;

	//8. 矩阵相乘、矩阵向量相乘
	std::cout << m*m << std::endl << std::endl;
	vec3d = Eigen::Vector3d(1, 2, 3);
	std::cout << m * vec3d << std::endl << std::endl;
	std::cout << vec3d.transpose()*m << std::endl << std::endl;

	//9. 矩阵的块操作
	std::cout << m << std::endl << std::endl;
	std::cout << m.block(1, 1, 2, 2) << std::endl << std::endl;
	std::cout << m.block<1, 2>(0, 0) << std::endl << std::endl;
	std::cout << m.col(1) << std::endl << std::endl;
	std::cout << m.row(0) << std::endl << std::endl;

	//10. 向量的块操作
	Eigen::ArrayXf arrayXf(10);
	arrayXf << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
	std::cout << vec3d << std::endl << std::endl;
	std::cout << arrayXf << std::endl << std::endl;
	std::cout << arrayXf.head(5) << std::endl << std::endl;
	std::cout << arrayXf.tail(4) * 2 << std::endl << std::endl;

	//11. 求解矩阵的特征值和特征向量
	Eigen::Matrix2f matrix2f;
	matrix2f << 1, 2, 3, 4;
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> eigenSolver(matrix2f);
	if (eigenSolver.info() == Eigen::Success) {
		std::cout << eigenSolver.eigenvalues() << std::endl << std::endl;
		std::cout << eigenSolver.eigenvectors() << std::endl << std::endl;
	}

	Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f>::RealVectorType eigen_values = eigenSolver.eigenvalues();
	Eigen::Vector2f vec = eigen_values.array();
	std::cout << vec[0] << "    " << vec[1] << std::endl;

	//12. 类Map及动态矩阵的使用
	int array1[4] = { 1, 2, 3, 4 };
	int array2[4] = { 5, 6, 7, 8 };
	int array3[4] = { 0, 0, 0, 0 };
	matrix_mul_matrix(array1, 2, 2, array2, 2, 2, array3);
	for (int i = 0; i < 4; i++)
		std::cout << array3[i] << std::endl;

	return 0;
}

int test_matrix_inverse()
{
	Eigen::MatrixXd m(3, 2), m_(2, 3);
	m << 1, 2, 3, 4, 5, 6;
	m_ = m.reverse(); // 转置矩阵

	Eigen::MatrixXd m2(3, 3), m2_(3, 3);
	m2 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
	m2_ = m2.inverse(); // 逆矩阵
	Eigen::MatrixXd m3 = m2.inverse();

	return 0;
}
