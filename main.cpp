//#include "multiplication_arrays.h"
//#include "multiplication_vectors.h"

#include <iostream>
#include <immintrin.h>
#include <chrono>
#include <random>
#include <thread>
#include <vector>
#include <mutex>
#include <string>

using namespace std;
using namespace std::chrono;

const extern unsigned int N_CONST = 4096;
extern unsigned int N = 1024;
extern const unsigned int random_number_limit_excl = 100;
const auto processor_count = std::thread::hardware_concurrency();
const uint64_t NUM_OF_ELEMENTS_IN_SUM = 10000000000;

void run_multiplication_vectors(FILE* file);
void generate_random_matrix_vectors(vector<int>& A, int N, int start_index);
bool matrices_are_equal_vectors(vector<int>& A, vector<int>& B, int N);
bool matrices_are_equal_vectors(vector<int>& A, vector<int>& B, vector<int>& C, vector<int>& D, int N);
void scalar_matr_mult_transposed_vectors_parallel(vector<int>& A, vector<int>& B, vector<int>& C, int starting_index, int finishing_index, int N);
void scalar_matr_mult_transposed_vectors(vector<int>& A, vector<int>& B, vector<int>& C, int N);
void transpose_vectors(vector<int>& A, int N);
void vector_matr_mult_transposed_vectors_parallel(vector<int>& A, vector<int>& B, vector<int>& C, int starting_index, int finishing_index, int N);
void vector_matr_mult_transposed_vectors(vector<int>& A, vector<int>& B, vector<int>& C, int N);

void run_multiplication_c_arrays(FILE* file);
void scalar_matr_mult_transposed_arrays(int** A, int** B, int** C, int N);
void vector_matr_mult_transposed_arrays(int** A, int** B, int** C, int N);
void scalar_matr_mult_transposed_arrays_parallel(int** A, int** B, int** C, int starting_index, int finishing_index, int N);
void vector_matr_mult_transposed_arrays_parallel(int** A, int** B, int** C, int starting_index, int finishing_index, int N);
bool matrices_are_equal_arrays(int** A, int** B, int N);
bool matrices_are_equal_arrays(int** A, int** B, int** C, int** D, int N);
void transpose_arrays(int** A, int N);
void delete_array(int** arr, int N);

void run_sum_calculation(FILE* file);
void run_sum_loop(uint64_t& sum);
void run_sum_loop_parallel(uint64_t& sum, uint64_t starting_index, uint64_t finishing_index);

int main()
{
	cout << "Calculating the sum of a series\n";
	string file_name1 = (string)"sum_output" + ".csv";
	FILE* output_file1 = fopen(file_name1.c_str(), "w");
	fprintf(output_file1, "no parallelism; parallelism\n");
	for (int q = 0; q < 1; q++)
	{
		run_sum_calculation(output_file1);
	}
	fclose(output_file1);

	cout << "Multiplying two matrices with dimensions of " << N << "x" << N << endl;
	string file_name2 = (string)"multiplication_output_optimized" + ".csv";
	FILE* output_file2 = fopen(file_name2.c_str(), "w");
	fprintf(output_file2, "array scalar; array simd; array scalar paral; array simd paral; vector scalar; vector simd; vector scalar paral; vector simd paral\n");
	for (int q = 1; q <= 1; q++)
	{
		run_multiplication_c_arrays(output_file2);
		run_multiplication_vectors(output_file2);
	}
	fclose(output_file2);

	return 0;
}

void run_sum_calculation(FILE* file)
{
	cout << "Results for sum calculation (no parallelism)" << endl;
	uint64_t sum1 = 0;
	auto start1 = high_resolution_clock::now();
	run_sum_loop(sum1);
	auto end1 = high_resolution_clock::now();
	auto duration1 = duration_cast<milliseconds>(end1 - start1);
	cout << "Time: " << duration1.count() << " milliseconds" << endl;
	fprintf(file, "%lld;", duration1.count());

	cout << "Results for sum calculation (parallelism)" << endl;
	uint64_t sum2 = 0;
	vector<uint64_t> sum_parts(processor_count);
	vector<thread> thrs1;

	auto start2 = high_resolution_clock::now();
	for (size_t i = 0; i < processor_count; i++)
	{
		thrs1.push_back(move(thread(run_sum_loop_parallel, ref(sum_parts[i]), NUM_OF_ELEMENTS_IN_SUM / processor_count * i, NUM_OF_ELEMENTS_IN_SUM / processor_count * (i + 1))));
	}
	for (std::thread& th : thrs1)
	{
		if (th.joinable()) th.join();	
	}
	for (int i = 0; i < processor_count; i++)
	{
		sum2 += sum_parts[i];
	}
	auto end2 = high_resolution_clock::now();
	auto duration2 = duration_cast<milliseconds>(end2 - start2);
	cout << "Time: " << duration2.count() << " milliseconds" << endl;
	fprintf(file, "%lld;\n", duration2.count());

	cout << "Results are equal: " << (sum1 == sum2) << "\n\n";

}

void run_sum_loop(uint64_t& sum)
{
	uint64_t sum_tmp = 0;
	for (uint64_t ind = 1; ind <= NUM_OF_ELEMENTS_IN_SUM; ind++)
	{
		// commented slows down the summation significantly
		// sum += ind;
		sum_tmp += ind;
	}
	sum += sum_tmp;
}

void run_sum_loop_parallel(uint64_t& sum, uint64_t starting_index, uint64_t finishing_index)
{
	uint64_t sum_tmp = 0;
	for (uint64_t ind = starting_index + 1; ind <= finishing_index; ind++)
	{
		// commented slows down the summation significantly
		// sum += ind;
		sum_tmp += ind;
	}
	sum += sum_tmp;
}

void run_multiplication_vectors(FILE* file)
{
	srand(time(0));

	cout << "Results for matrix multiplication using std::vector, SIMD and Parallelism\n";
	cout << "Creating vectors..." << endl;

	vector<int> A(N * N);
	vector<int> B(N * N);
	vector<int> scalar_result(N * N);
	vector<int> vector_result(N * N);
	vector<int> scalar_result_par(N * N);
	vector<int> vector_result_par(N * N);

	cout << "Filling arrays with random numbers..." << endl;

	generate_random_matrix_vectors(A, N * N, 0);
	generate_random_matrix_vectors(B, N * N, 0);

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			scalar_result[i * N + j] = 0;
			vector_result[i * N + j] = 0;
			scalar_result_par[i * N + j] = 0;
			vector_result_par[i * N + j] = 0;
		}
	}

	transpose_vectors(B, N);

	cout << "Multiplying matrices (scalar, no parallelism)..." << endl;
	auto start1 = high_resolution_clock::now();
	scalar_matr_mult_transposed_vectors(A, B, scalar_result, N);
	auto end1 = high_resolution_clock::now();
	auto duration1 = duration_cast<milliseconds>(end1 - start1);
	cout << "Time: " << duration1.count() << " milliseconds" << endl;

	cout << "Multiplying matrices (AVX), no parallelism..." << endl;
	auto start2 = high_resolution_clock::now();
	vector_matr_mult_transposed_vectors(A, B, vector_result, N);
	auto end2 = high_resolution_clock::now();
	auto duration2 = duration_cast<milliseconds>(end2 - start2);
	cout << "Time: " << duration2.count() << " milliseconds" << endl;

	/*cout << "Results are equal: " << matrices_are_equal_vectors(scalar_result, vector_result, N) << endl;*/

	cout << "Multiplying matrices (scalar, yes parallelism)..." << endl;
	auto start3 = high_resolution_clock::now();
	vector<thread> thrs1;
	for (size_t i = 0; i < processor_count; i++)
	{
		thrs1.push_back(move(thread(scalar_matr_mult_transposed_vectors_parallel, ref(A), ref(B), ref(scalar_result_par), N / processor_count * i, N / processor_count * (i + 1), N)));
	}
	for (std::thread& th : thrs1)
	{
		if (th.joinable()) th.join();
	}
	auto end3 = high_resolution_clock::now();
	auto duration3 = duration_cast<milliseconds>(end3 - start3);
	cout << "Time: " << duration3.count() << " milliseconds" << endl;

	cout << "Multiplying matrices (AVX), yes parallelism..." << endl;
	auto start4 = high_resolution_clock::now();
	vector<thread> thrs2;
	for (size_t i = 0; i < processor_count; i++)
	{
		thrs2.push_back(move(thread(vector_matr_mult_transposed_vectors_parallel, ref(A), ref(B), ref(vector_result_par), N / processor_count * i, N / processor_count * (i + 1), N)));
	}
	for (std::thread& th : thrs2)
	{
		if (th.joinable()) th.join();
	}
	auto end4 = high_resolution_clock::now();
	auto duration4 = duration_cast<milliseconds>(end4 - start4);
	cout << "Time: " << duration4.count() << " milliseconds" << endl;

	fprintf(file, "%lld;%lld;%lld;%lld\n", duration1.count(), duration2.count(), duration3.count(), duration4.count());

	cout << "Results are equal: " << matrices_are_equal_vectors(scalar_result, vector_result, scalar_result_par, vector_result_par, N) << "\n\n";

	// эксплицитно вызываем деструкторы
	A.~vector();
	B.~vector();
	scalar_result.~vector();
	vector_result.~vector();
	scalar_result_par.~vector();
	vector_result_par.~vector();
}

void generate_random_matrix_vectors(vector<int>& A, int N, int start_index)
{
	srand(time(0));
	for (int i = 0; i < N; i++)
	{
		A[i] = rand() % random_number_limit_excl;
	}
}

void scalar_matr_mult_transposed_vectors(vector<int>& A, vector<int>& B, vector<int>& C, int N)
{
	//cout << "Scalar multiplication (transpose): " << endl;
	for (int i = 0; i < N; i++)
	{
		//cout << "row " << i << endl;
		for (int j = 0; j < N; j++)
		{
			//cout << "col " << j << endl;
			for (int k = 0; k < N; k++)
			{
				C[i * N + j] += A[i * N + k] * B[j * N + k];
				//cout << "elem one: " << A[i][k] << ", elem two: " << B[k][j] << endl;
			}
		}
	}
}

void scalar_matr_mult_transposed_vectors_parallel(vector<int>& A, vector<int>& B, vector<int>& C, int starting_index, int finishing_index, int N)
{
	//cout << "Scalar multiplication (transpose): " << endl;
	for (int i = starting_index; i < finishing_index; i++)
	{
		//cout << "row " << i << endl;
		for (int j = 0; j < N; j++)
		{
			//cout << "col " << j << endl;
			for (int k = 0; k < N; k++)
			{
				C[i * N + j] += A[i * N + k] * B[j * N + k];
				//cout << "elem one: " << A[i][k] << ", elem two: " << B[k][j] << endl;
			}
		}
	}
}

void transpose_vectors(vector<int>& A, int N)
{
	vector<int> tmp(N * N);

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			tmp[i * N + j] = A[i * N + j];
		}
	}

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			A[j * N + i] = tmp[i * N + j];
		}
	}

	tmp.~vector();
}


bool matrices_are_equal_vectors(vector<int>& A, vector<int>& B, int N)
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			if (A[i * N + j] != B[i * N + j]) return 0;
		}
	}
	return 1;
}

bool matrices_are_equal_vectors(vector<int>& A, vector<int>& B, vector<int>& C, vector<int>& D, int N)
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			if (A[i * N + j] != B[i * N + j] || A[i * N + j] != C[i * N + j] || A[i * N + j] != D[i * N + j]
				|| B[i * N + j] != C[i * N + j] || C[i * N + j] != D[i * N + j]) return 0;
		}
	}
	return 1;
}

void vector_matr_mult_transposed_vectors(vector<int>& A, vector<int>& B, vector<int>& C, int N)
{
	__m256i input_left;
	__m256i input_right;
	__m256i output_tmp;
	int output_final;

	for (int i = 0; i < N; i++)
	{
		//cout << "row " << i << endl;
		for (int j = 0; j < N; j++)
		{
			output_final = 0;
			//cout << "col " << j << endl;
			for (int k = 0; k < N; k += 8)
			{
				// загружаем в память 256 байт целочисленных данных
				input_left = _mm256_load_si256((__m256i*) & A[i * N + k]);
				input_right = _mm256_load_si256((__m256i*) & B[j * N + k]);
				// поэлементно перемножаем два вектора, в результате получается временные 64-битные целочисленные переменные
				// нижние 32 бита этих переменных храним в output_tmp
				output_tmp = _mm256_mullo_epi32(input_left, input_right);
				// числа, полученные в результате умножения, прибавляем к сумме
				output_final += /*output_tmp.m256i_i32[0] + output_tmp.m256i_i32[1] +
					output_tmp.m256i_i32[2] + output_tmp.m256i_i32[3] +
					output_tmp.m256i_i32[4] + output_tmp.m256i_i32[5] +
					output_tmp.m256i_i32[6] + output_tmp.m256i_i32[7];*/
					_mm256_extract_epi32(output_tmp, 0) + _mm256_extract_epi32(output_tmp, 1)
					+ _mm256_extract_epi32(output_tmp, 2) + _mm256_extract_epi32(output_tmp, 3)
					+ _mm256_extract_epi32(output_tmp, 4) + _mm256_extract_epi32(output_tmp, 5)
					+ _mm256_extract_epi32(output_tmp, 6) + _mm256_extract_epi32(output_tmp, 7);
			}
			// прошли всю строку, получили финальное значение элемента
			C[i * N + j] = output_final;
		}
	}
}

void vector_matr_mult_transposed_vectors_parallel(vector<int>& A, vector<int>& B, vector<int>& C, int starting_index, int finishing_index, int N)
{
	__m256i input_left;
	__m256i input_right;
	__m256i output_tmp;
	int output_final;

	for (int i = starting_index; i < finishing_index; i++)
	{
		//cout << "row " << i << endl;
		for (int j = 0; j < N; j++)
		{
			output_final = 0;
			//cout << "col " << j << endl;
			for (int k = 0; k < N; k += 8)
			{
				// загружаем в память 256 байт целочисленных данных
				input_left = _mm256_load_si256((__m256i*) & A[i * N + k]);
				input_right = _mm256_load_si256((__m256i*) & B[j * N + k]);
				// поэлементно перемножаем два вектора, в результате получается временные 64-битные целочисленные переменные
				// нижние 32 бита этих переменных храним в output_tmp
				output_tmp = _mm256_mullo_epi32(input_left, input_right);
				// числа, полученные в результате умножения, прибавляем к сумме
				output_final += /*output_tmp.m256i_i32[0] + output_tmp.m256i_i32[1] +
					output_tmp.m256i_i32[2] + output_tmp.m256i_i32[3] +
					output_tmp.m256i_i32[4] + output_tmp.m256i_i32[5] +
					output_tmp.m256i_i32[6] + output_tmp.m256i_i32[7];*/
					_mm256_extract_epi32(output_tmp, 0) + _mm256_extract_epi32(output_tmp, 1)
					+ _mm256_extract_epi32(output_tmp, 2) + _mm256_extract_epi32(output_tmp, 3)
					+ _mm256_extract_epi32(output_tmp, 4) + _mm256_extract_epi32(output_tmp, 5)
					+ _mm256_extract_epi32(output_tmp, 6) + _mm256_extract_epi32(output_tmp, 7);
			}
			// прошли всю строку, получили финальное значение элемента
			C[i * N + j] = output_final;
		}
	}
}

void run_multiplication_c_arrays(FILE* file)
{
	srand(time(0));

	cout << "Results for matrix multiplication using C-style arrays, SIMD and parallelism\n";
	cout << "Creating C arrays..." << endl;

	int** A = new int* [N]; // вход 1
	int** B = new int* [N]; // вход 2
	int** scalar_result = new int* [N]; // выход (скаляр)
	int** vector_result = new int* [N]; // выход (вектор)
	int** scalar_result_par = new int* [N]; // выход (скаляр, параллелизм)
	int** vector_result_par = new int* [N]; // выход (вектор, паралеллизм)

	for (int i = 0; i < N; i++)
	{
		A[i] = new int[N];
		B[i] = new int[N];
		scalar_result[i] = new int[N];
		vector_result[i] = new int[N];
		scalar_result_par[i] = new int[N];
		vector_result_par[i] = new int[N];
	}

	cout << "Filling array with random numbers..." << endl;

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			A[i][j] = rand() % random_number_limit_excl;
			B[i][j] = rand() % random_number_limit_excl;
			scalar_result[i][j] = 0;
			vector_result[i][j] = 0;
			scalar_result_par[i][j] = 0;
			vector_result_par[i][j] = 0;
		}
	}

	transpose_arrays(B, N);

	cout << "Multiplying matrices (scalar, no parallelism)..." << endl;
	auto start1 = high_resolution_clock::now();
	scalar_matr_mult_transposed_arrays(A, B, scalar_result, N);
	auto end1 = high_resolution_clock::now();
	auto duration1 = duration_cast<milliseconds>(end1 - start1);
	cout << "Time: " << duration1.count() << " milliseconds" << endl;

	cout << "Multiplying matrices (AVX), no parallelism..." << endl;
	auto start2 = high_resolution_clock::now();
	vector_matr_mult_transposed_arrays(A, B, vector_result, N);
	auto end2 = high_resolution_clock::now();
	auto duration2 = duration_cast<milliseconds>(end2 - start2);
	cout << "Time: " << duration2.count() << " milliseconds" << endl;

	//cout << "Results are equal: " << matrices_are_equal_arrays(scalar_result, vector_result, N) << endl;

	cout << "Multiplying matrices (scalar, yes parallelism)..." << endl;
	auto start3 = high_resolution_clock::now();
	vector<thread> thrs1;
	for (size_t i = 0; i < processor_count; i++)
	{
		thrs1.push_back(move(thread(scalar_matr_mult_transposed_arrays_parallel, A, B, scalar_result_par, N / processor_count * i, N / processor_count * (i + 1), N)));
	}
	for (std::thread& th : thrs1)
	{
		if (th.joinable()) th.join();
	}
	auto end3 = high_resolution_clock::now();
	auto duration3 = duration_cast<milliseconds>(end3 - start3);
	cout << "Time: " << duration3.count() << " milliseconds" << endl;

	cout << "Multiplying matrices (AVX), yes parallelism..." << endl;
	auto start4 = high_resolution_clock::now();
	vector<thread> thrs2;
	for (size_t i = 0; i < processor_count; i++)
	{
		thrs2.push_back(move(thread(vector_matr_mult_transposed_arrays_parallel, A, B, vector_result_par, N / processor_count * i, N / processor_count * (i + 1), N)));
	}
	for (std::thread& th : thrs2)
	{
		if (th.joinable()) th.join();
	}
	auto end4 = high_resolution_clock::now();
	auto duration4 = duration_cast<milliseconds>(end4 - start4);
	cout << "Time: " << duration4.count() << " milliseconds" << endl;

	fprintf(file, "%lld;%lld;%lld;%lld;", duration1.count(), duration2.count(), duration3.count(), duration4.count());

	cout << "Results are equal: " << matrices_are_equal_arrays(scalar_result, vector_result, scalar_result_par, vector_result_par, N) << "\n\n";

	delete_array(A, N);
	delete_array(B, N);
	delete_array(scalar_result, N);
	delete_array(vector_result, N);
	delete_array(scalar_result_par, N);
	delete_array(vector_result_par, N);
}


void scalar_matr_mult_transposed_arrays(int** A, int** B, int** C, int N)
{
	//cout << "Scalar multiplication (transpose): " << endl;
	for (int i = 0; i < N; i++)
	{
		//cout << "row " << i << endl;
		for (int j = 0; j < N; j++)
		{
			//cout << "col " << j << endl;
			for (int k = 0; k < N; k++)
			{
				C[i][j] += A[i][k] * B[j][k];
				//cout << "elem one: " << A[i][k] << ", elem two: " << B[k][j] << endl;
			}
		}
	}
}

void scalar_matr_mult_transposed_arrays_parallel(int** A, int** B, int** C, int starting_index, int finishing_index, int N)
{
	//cout << "Scalar multiplication (transpose): " << endl;
	for (int i = starting_index; i < finishing_index; i++)
	{
		//cout << "row " << i << endl;
		for (int j = 0; j < N; j++)
		{
			//cout << "col " << j << endl;
			for (int k = 0; k < N; k++)
			{
				C[i][j] += A[i][k] * B[j][k];
				//cout << "elem one: " << A[i][k] << ", elem two: " << B[k][j] << endl;
			}
		}
	}
}

void transpose_arrays(int** A, int N)
{
	int** tmp = new int* [N];
	for (int i = 0; i < N; i++)
	{
		tmp[i] = new int[N];
	}

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			tmp[i][j] = A[i][j];
		}
	}

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			A[j][i] = tmp[i][j];
		}
	}

	delete_array(tmp, N);
}

void delete_array(int** arr, int N)
{
	for (int i = 0; i < N; i++)
	{
		delete[] arr[i];
	}
	delete[] arr;
}

bool matrices_are_equal_arrays(int** A, int** B, int N)
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			if (A[i][j] != B[i][j]) return 0;
		}
	}
	return 1;
}

bool matrices_are_equal_arrays(int** A, int** B, int** C, int** D, int N)
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			if (A[i][j] != B[i][j] || A[i][j] != C[i][j] || A[i][j] != D[i][j]
				|| B[i][j] != C[i][j] || B[i][j] != D[i][j] || C[i][j] != D[i][j]) return 0;
		}
	}
	return 1;
}

// исходим из того, что вторая матрица транспонированная
void vector_matr_mult_transposed_arrays(int** A, int** B, int** C, int N)
{
	__m256i input_left;
	__m256i input_right;
	__m256i output_tmp;
	int output_final;

	for (int i = 0; i < N; i++)
	{
		//cout << "row " << i << endl;
		for (int j = 0; j < N; j++)
		{
			output_final = 0;
			//cout << "col " << j << endl;
			for (int k = 0; k < N; k += 8)
			{
				// загружаем в память 256 байт целочисленных данных
				input_left = _mm256_load_si256((__m256i*) & A[i][k]);
				input_right = _mm256_load_si256((__m256i*) & B[j][k]);
				// поэлементно перемножаем два вектора, в результате получается временные 64-битные целочисленные переменные
				// нижние 32 бита этих переменных храним в output_tmp
				output_tmp = _mm256_mullo_epi32(input_left, input_right);
				// числа, полученные в результате умножения, прибавляем к сумме
				output_final +=
					_mm256_extract_epi32(output_tmp, 0) + _mm256_extract_epi32(output_tmp, 1)
					+ _mm256_extract_epi32(output_tmp, 2) + _mm256_extract_epi32(output_tmp, 3)
					+ _mm256_extract_epi32(output_tmp, 4) + _mm256_extract_epi32(output_tmp, 5)
					+ _mm256_extract_epi32(output_tmp, 6) + _mm256_extract_epi32(output_tmp, 7);
			}
			// прошли всю строку, получили финальное значение элемента
			C[i][j] = output_final;
		}
	}
}

void vector_matr_mult_transposed_arrays_parallel(int** A, int** B, int** C, int starting_index, int finishing_index, int N)
{
	__m256i input_left;
	__m256i input_right;
	__m256i output_tmp;
	int output_final;

	for (int i = starting_index; i < finishing_index; i++)
	{
		//cout << "row " << i << endl;
		for (int j = 0; j < N; j++)
		{
			output_final = 0;
			//cout << "col " << j << endl;
			for (int k = 0; k < N; k += 8)
			{
				// загружаем в память 256 байт целочисленных данных
				input_left = _mm256_load_si256((__m256i*) & A[i][k]);
				input_right = _mm256_load_si256((__m256i*) & B[j][k]);
				// поэлементно перемножаем два вектора, в результате получается временные 64-битные целочисленные переменные
				// нижние 32 бита этих переменных храним в output_tmp
				output_tmp = _mm256_mullo_epi32(input_left, input_right);
				// числа, полученные в результате умножения, прибавляем к сумме
				output_final +=
					_mm256_extract_epi32(output_tmp, 0) + _mm256_extract_epi32(output_tmp, 1)
					+ _mm256_extract_epi32(output_tmp, 2) + _mm256_extract_epi32(output_tmp, 3)
					+ _mm256_extract_epi32(output_tmp, 4) + _mm256_extract_epi32(output_tmp, 5)
					+ _mm256_extract_epi32(output_tmp, 6) + _mm256_extract_epi32(output_tmp, 7);
			}
			// прошли всю строку, получили финальное значение элемента
			C[i][j] = output_final;
		}
	}
}
