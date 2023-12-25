#ifndef __CUMAT_MATRIX_BASE_H__
#define __CUMAT_MATRIX_BASE_H__

#include <cuda_runtime.h>

#include "Macros.h"
#include "ForwardDeclarations.h"
#include "Constants.h"

CUMAT_NAMESPACE_BEGIN

/**
 * \brief The base class of all matrix types and matrix expressions.
 * \tparam _Derived
 */
template<typename _Derived>
class MatrixBase
{
public:
	typedef _Derived Type;
	typedef MatrixBase<_Derived> Base;
	CUMAT_PUBLIC_API_NO_METHODS;

	/**
	 * \returns a reference to the _Derived object
	 */
	__host__ __device__ CUMAT_STRONG_INLINE _Derived& derived() { return *static_cast<_Derived*>(this); }

	/**
	 * \returns a const reference to the _Derived object
	 */
	__host__ __device__ CUMAT_STRONG_INLINE const _Derived& derived() const { return *static_cast<const _Derived*>(this); }

	/**
	 * \brief Returns the number of rows of this matrix.
	 * \returns the number of rows.
	 */
	__host__ __device__ CUMAT_STRONG_INLINE Index rows() const { return derived().rows(); }
	/**
	 * \brief Returns the number of columns of this matrix.
	 * \returns the number of columns.
	 */
	__host__ __device__ CUMAT_STRONG_INLINE Index cols() const { return derived().cols(); }
	/**
	 * \brief Returns the number of batches of this matrix.
	 * \returns the number of batches.
	 */
	__host__ __device__ CUMAT_STRONG_INLINE Index batches() const { return derived().batches(); }
	/**
	* \brief Returns the total number of entries in this matrix.
	* This value is computed as \code rows()*cols()*batches()* \endcode
	* \return the total number of entries
	*/
	__host__ __device__ CUMAT_STRONG_INLINE Index size() const { return rows() * cols() * batches(); }

	// EVALUATION

	typedef Matrix<
		typename internal::traits<_Derived>::Scalar,
		internal::traits<_Derived>::RowsAtCompileTime,
		internal::traits<_Derived>::ColsAtCompileTime,
		internal::traits<_Derived>::BatchesAtCompileTime,
		internal::traits<_Derived>::Flags
	> eval_t;

	/**
	 * \brief Evaluates this into a matrix.
	 * This evaluates any expression template. If this is already a matrix, it is returned unchanged.
	 * \return the evaluated matrix
	 */
	eval_t eval() const
	{
		return eval_t(derived());
	}

	/**
	 * \brief Conversion: Matrix of size 1-1-1 (scalar) in device memory to the host memory scalar.
	 *
	 * This is expecially usefull to directly use the results of full reductions in host code.
	 *
	 * \tparam T
	 */
	explicit operator Scalar () const
	{
		CUMAT_STATIC_ASSERT(
			internal::traits<_Derived>::RowsAtCompileTime == 1 &&
			internal::traits<_Derived>::ColsAtCompileTime == 1 &&
			internal::traits<_Derived>::BatchesAtCompileTime == 1,
			"Conversion only possible for compile-time scalars");
		eval_t m = eval();
		Scalar v;
		m.copyToHost(&v);
		return v;
	}


	// CWISE EXPRESSIONS
	// #include "MatrixBlockPluginRvalue.inl"

	//most general version, static size

	/**
	 * \brief Creates a block of the matrix of static size.
	 * By using this method, you can convert a dynamically-sized matrix into a statically sized one.
	 *
	 * \param start_row the start row of the block (zero based)
	 * \param start_column the start column of the block (zero based)
	 * \param start_batch the start batch of the block (zero based)
	 * \tparam NRows the number of rows of the block on compile time
	 * \tparam NColumsn the number of columns of the block on compile time
	 * \tparam NBatches the number of batches of the block on compile time
	 */
	template<int NRows, int NColumns, int NBatches>
	MatrixBlock<typename internal::traits<_Derived>::Scalar, NRows, NColumns, NBatches, internal::traits<_Derived>::Flags, const _Derived>
		block(Index start_row, Index start_column, Index start_batch, Index num_rows = NRows,
			Index num_columns = NColumns, Index num_batches = NBatches) const
	{
		CUMAT_ERROR_IF_NO_NVCC(block)
			CUMAT_ASSERT_ARGUMENT(NRows > 0 ? NRows == num_rows : true);
		CUMAT_ASSERT_ARGUMENT(NColumns > 0 ? NColumns == num_columns : true);
		CUMAT_ASSERT_ARGUMENT(NBatches > 0 ? NBatches == num_batches : true);
		CUMAT_ASSERT_ARGUMENT(num_rows >= 0);
		CUMAT_ASSERT_ARGUMENT(num_columns >= 0);
		CUMAT_ASSERT_ARGUMENT(num_batches >= 0);
		CUMAT_ASSERT_ARGUMENT(start_row >= 0);
		CUMAT_ASSERT_ARGUMENT(start_column >= 0);
		CUMAT_ASSERT_ARGUMENT(start_batch >= 0);
		CUMAT_ASSERT_ARGUMENT(start_row + num_rows <= rows());
		CUMAT_ASSERT_ARGUMENT(start_column + num_columns <= cols());
		CUMAT_ASSERT_ARGUMENT(start_batch + num_batches <= batches());
		return MatrixBlock<typename internal::traits<_Derived>::Scalar, NRows, NColumns, NBatches, internal::traits<_Derived>::Flags, const _Derived>(
			derived(), num_rows, num_columns, num_batches, start_row, start_column, start_batch);
	}

	//most general version, dynamic size

	/**
	* \brief Creates a block of the matrix of dynamic size.
	*
	* \param start_row the start row of the block (zero based)
	* \param start_column the start column of the block (zero based)
	* \param start_batch the start batch of the block (zero based)
	* \param num_rows the number of rows in the block
	* \param num_columns the number of columns in the block
	* \param num_batches the number of batches in the block
	*/
	MatrixBlock<typename internal::traits<_Derived>::Scalar, Dynamic, Dynamic, Dynamic, internal::traits<_Derived>::Flags, const _Derived>
		block(Index start_row, Index start_column, Index start_batch, Index num_rows, Index num_columns, Index num_batches) const
	{
		CUMAT_ERROR_IF_NO_NVCC(block)
			CUMAT_ASSERT_ARGUMENT(start_row >= 0);
		CUMAT_ASSERT_ARGUMENT(start_column >= 0);
		CUMAT_ASSERT_ARGUMENT(start_batch >= 0);
		CUMAT_ASSERT_ARGUMENT(num_rows > 0);
		CUMAT_ASSERT_ARGUMENT(num_columns > 0);
		CUMAT_ASSERT_ARGUMENT(num_batches > 0);
		CUMAT_ASSERT_ARGUMENT(start_row + num_rows <= rows());
		CUMAT_ASSERT_ARGUMENT(start_column + num_columns <= cols());
		CUMAT_ASSERT_ARGUMENT(start_batch + num_batches <= batches());
		return MatrixBlock<typename internal::traits<_Derived>::Scalar, Dynamic, Dynamic, Dynamic, internal::traits<_Derived>::Flags, const _Derived>(
			derived(), num_rows, num_columns, num_batches, start_row, start_column, start_batch);
	}

	// specializations for batch==1, vectors, slices

	/**
	* \brief Extracts a row out of the matrix.
	* \param row the index of the row
	*/
	MatrixBlock<
		typename internal::traits<_Derived>::Scalar,
		1, internal::traits<_Derived>::ColsAtCompileTime, internal::traits<_Derived>::BatchesAtCompileTime,
		internal::traits<_Derived>::Flags, const _Derived>
		row(Index row) const
	{
		CUMAT_ERROR_IF_NO_NVCC(row)
			CUMAT_ASSERT_ARGUMENT(row >= 0);
		CUMAT_ASSERT_ARGUMENT(row < rows());
		return MatrixBlock<
			typename internal::traits<_Derived>::Scalar,
			1, internal::traits<_Derived>::ColsAtCompileTime, internal::traits<_Derived>::BatchesAtCompileTime,
			internal::traits<_Derived>::Flags, const _Derived>(
				derived(), 1, cols(), batches(), row, 0, 0);
	}

	/**
	* \brief Extracts a column out of the matrix.
	* \param col the index of the column
	*/
	MatrixBlock<
		typename internal::traits<_Derived>::Scalar,
		internal::traits<_Derived>::RowsAtCompileTime, 1, internal::traits<_Derived>::BatchesAtCompileTime,
		internal::traits<_Derived>::Flags, const _Derived>
		col(Index column) const
	{
		CUMAT_ERROR_IF_NO_NVCC(col)
			CUMAT_ASSERT_ARGUMENT(column >= 0);
		CUMAT_ASSERT_ARGUMENT(column < cols());
		return MatrixBlock<
			typename internal::traits<_Derived>::Scalar,
			internal::traits<_Derived>::RowsAtCompileTime, 1, internal::traits<_Derived>::BatchesAtCompileTime,
			internal::traits<_Derived>::Flags, const _Derived>(
				derived(), rows(), 1, batches(), 0, column, 0);
	}

	/**
	* \brief Extracts a slice of a specific batch out of the batched matrix
	* \param batch the index of the batch
	*/
	MatrixBlock<
		typename internal::traits<_Derived>::Scalar,
		internal::traits<_Derived>::RowsAtCompileTime, internal::traits<_Derived>::ColsAtCompileTime, 1,
		internal::traits<_Derived>::Flags, const _Derived>
		slice(Index batch) const
	{
		CUMAT_ERROR_IF_NO_NVCC(slice)
			CUMAT_ASSERT_ARGUMENT(batch >= 0);
		CUMAT_ASSERT_ARGUMENT(batch < batches());
		return MatrixBlock<
			typename internal::traits<_Derived>::Scalar,
			internal::traits<_Derived>::RowsAtCompileTime, internal::traits<_Derived>::ColsAtCompileTime, 1,
			internal::traits<_Derived>::Flags, const _Derived>(
				derived(), rows(), cols(), 1, 0, 0, batch);
	}


	// Vector operations

private:
	template<int N>
	MatrixBlock<
		typename internal::traits<_Derived>::Scalar,
		N,
		1,
		internal::traits<_Derived>::BatchesAtCompileTime,
		internal::traits<_Derived>::Flags, const _Derived>
		segmentHelper(Index start, std::true_type) const
	{
		//column vector
		CUMAT_ASSERT_ARGUMENT(start >= 0);
		CUMAT_ASSERT_ARGUMENT(start + N <= rows());
		return MatrixBlock<
			typename internal::traits<_Derived>::Scalar,
			N, 1, internal::traits<_Derived>::BatchesAtCompileTime,
			internal::traits<_Derived>::Flags, const _Derived>(
				derived(), N, 1, batches(), start, 0, 0);
	}

	template<int N>
	MatrixBlock<
		typename internal::traits<_Derived>::Scalar,
		1,
		N,
		internal::traits<_Derived>::BatchesAtCompileTime,
		internal::traits<_Derived>::Flags, const _Derived>
		segmentHelper(Index start, std::false_type) const
	{
		//row vector
		CUMAT_ASSERT_ARGUMENT(start >= 0);
		CUMAT_ASSERT_ARGUMENT(start + N <= cols());
		return MatrixBlock<
			typename internal::traits<_Derived>::Scalar,
			1, N, internal::traits<_Derived>::BatchesAtCompileTime,
			internal::traits<_Derived>::Flags, const _Derived>(
				derived(), 1, N, batches(), 0, start, 0);
	}

public:

	/**
	* \brief Extracts a fixed-size segment of the vector.
	*   Only available for vectors
	* \param start the start position of the segment
	* \tparam N the length of the segment
	*/
	template<int N>
	auto //FixedVectorSegmentXpr<N>::Type
		segment(Index start) const -> decltype(segmentHelper<N>(start, std::integral_constant<bool, internal::traits<_Derived>::ColsAtCompileTime == 1>()))
	{
		CUMAT_ERROR_IF_NO_NVCC(segment)
			CUMAT_STATIC_ASSERT(
				(internal::traits<_Derived>::RowsAtCompileTime == 1 || internal::traits<_Derived>::ColsAtCompileTime == 1),
				"segment can only act on compile-time vectors");
		return segmentHelper<N>(start, std::integral_constant<bool, internal::traits<_Derived>::ColsAtCompileTime == 1>());
	}

	/**
	 * \brief Extracts a fixed-size segment from the head of the vector.
	 * Only available for vectors
	 * \tparam N the length of the segment
	 */
	template<int N>
	auto head() const -> decltype(segment<N>(0))
	{
		CUMAT_ERROR_IF_NO_NVCC(head)
			return segment<N>(0);
	}

	/**
	* \brief Extracts a fixed-size segment from the tail of the vector.
	* Only available for vectors
	* \tparam N the length of the segment
	*/
	template<int N>
	auto tail() const -> decltype(segment<N>(0))
	{
		CUMAT_ERROR_IF_NO_NVCC(tail)
			return segment<N>(std::max(rows(), cols()) - N);
	}

private:
	MatrixBlock<
		typename internal::traits<_Derived>::Scalar,
		Dynamic,
		1,
		internal::traits<_Derived>::BatchesAtCompileTime,
		internal::traits<_Derived>::Flags, const _Derived>
		segmentHelper(Index start, Index length, std::true_type) const
	{
		//column vector
		CUMAT_ASSERT_ARGUMENT(start >= 0);
		CUMAT_ASSERT_ARGUMENT(start + length <= rows());
		return MatrixBlock<
			typename internal::traits<_Derived>::Scalar,
			Dynamic, 1, internal::traits<_Derived>::BatchesAtCompileTime,
			internal::traits<_Derived>::Flags, const _Derived>(
				derived(), length, 1, batches(), start, 0, 0);
	}
	MatrixBlock<
		typename internal::traits<_Derived>::Scalar,
		1,
		Dynamic,
		internal::traits<_Derived>::BatchesAtCompileTime,
		internal::traits<_Derived>::Flags, const _Derived>
		segmentHelper(Index start, Index length, std::false_type) const
	{
		//row vector
		CUMAT_ASSERT_ARGUMENT(start >= 0);
		CUMAT_ASSERT_ARGUMENT(start + length <= cols());
		return MatrixBlock<
			typename internal::traits<_Derived>::Scalar,
			1, Dynamic, internal::traits<_Derived>::BatchesAtCompileTime,
			internal::traits<_Derived>::Flags, const _Derived>(
				derived(), 1, length, batches(), 0, start, 0);
	}

public:
	/**
	* \brief Extracts a dynamic-size segment of the vector.
	*   Only available for vectors
	* \param start the start position of the segment
	* \param length the length of the segment
	*/
	auto
		segment(Index start, Index length) const -> decltype(segmentHelper(start, length, std::integral_constant<bool, internal::traits<_Derived>::ColsAtCompileTime == 1>()))
	{
		CUMAT_ERROR_IF_NO_NVCC(segment)
			CUMAT_STATIC_ASSERT(
				(internal::traits<_Derived>::RowsAtCompileTime == 1 || internal::traits<_Derived>::ColsAtCompileTime == 1),
				"segment can only act on compile-time vectors");
		return segmentHelper(start, length, std::integral_constant<bool, internal::traits<_Derived>::ColsAtCompileTime == 1>());
	}

	/**
	* \brief Extracts a dynamic-size segment from the head of the vector.
	* Only available for vectors
	* \param length the length of the segment
	*/
	auto head(Index length) const -> decltype(segment(0, length))
	{
		CUMAT_ERROR_IF_NO_NVCC(head)
			return segment(0, length);
	}

	/**
	* \brief Extracts a dynamic-size segment from the tail of the vector.
	* Only available for vectors
	* \param length the length of the segment
	*/
	auto tail(Index length) const -> decltype(segment(0, length))
	{
		CUMAT_ERROR_IF_NO_NVCC(tail)
			return segment(std::max(rows(), cols()) - length, length);
	}


	//#include "UnaryOpsPlugin.inl"
	//Included inside MatrixBase, define the accessors

#define UNARY_OP_ACCESSOR(Name) \
	UnaryOp<_Derived, functor::UnaryMathFunctor_ ## Name <Scalar> > Name () const { \
		CUMAT_ERROR_IF_NO_NVCC(Name)  \
		return UnaryOp<_Derived, functor::UnaryMathFunctor_ ## Name <Scalar> >(derived()); \
	}

/**
 * \brief computes the component-wise negation (x -> -x)
 */
	UNARY_OP_ACCESSOR(cwiseNegate);
	/**
	* \brief computes the component-wise absolute value (x -> |x|)
	*/
	UNARY_OP_ACCESSOR(cwiseAbs);
	/**
	* \brief squares the value (x -> x^2)
	*/
	UNARY_OP_ACCESSOR(cwiseAbs2);
	/**
	* \brief computes the component-wise inverse (x -> 1/x)
	*/
	UNARY_OP_ACCESSOR(cwiseInverse);
	/**
	* \brief computes the component-wise inverse (x -> 1/x)
	* with the additional check that it returns 1 if x is zero.
	*/
	UNARY_OP_ACCESSOR(cwiseInverseCheck);

	/**
	* \brief computes the component-wise exponent (x -> exp(x))
	*/
	UNARY_OP_ACCESSOR(cwiseExp);
	/**
	* \brief computes the component-wise natural logarithm (x -> log(x))
	*/
	UNARY_OP_ACCESSOR(cwiseLog);
	/**
	* \brief computes the component-wise value of (x -> log(x+1))
	*/
	UNARY_OP_ACCESSOR(cwiseLog1p);
	/**
	* \brief computes the component-wise value of (x -> log_10(x))
	*/
	UNARY_OP_ACCESSOR(cwiseLog10);

	/**
	* \brief computes the component-wise square root (x -> sqrt(x))
	*/
	UNARY_OP_ACCESSOR(cwiseSqrt);
	/**
	* \brief computes the component-wise reciprocal square root (x -> 1 / sqrt(x))
	*/
	UNARY_OP_ACCESSOR(cwiseRsqrt);
	/**
	* \brief computes the component-wise cube root (x -> x^(1/3))
	*/
	UNARY_OP_ACCESSOR(cwiseCbrt);
	/**
	* \brief computes the component-wise reciprocal cube root (x -> x^(-1/3))
	*/
	UNARY_OP_ACCESSOR(cwiseRcbrt);

	/**
	* \brief computes the component-wise value of (x -> sin(x))
	*/
	UNARY_OP_ACCESSOR(cwiseSin);
	/**
	* \brief computes the component-wise value of (x -> cos(x))
	*/
	UNARY_OP_ACCESSOR(cwiseCos);
	/**
	* \brief computes the component-wise value of (x -> tan(x))
	*/
	UNARY_OP_ACCESSOR(cwiseTan);
	/**
	* \brief computes the component-wise value of (x -> asin(x))
	*/
	UNARY_OP_ACCESSOR(cwiseAsin);
	/**
	* \brief computes the component-wise value of (x -> acos(x))
	*/
	UNARY_OP_ACCESSOR(cwiseAcos);
	/**
	* \brief computes the component-wise value of (x -> atan(x))
	*/
	UNARY_OP_ACCESSOR(cwiseAtan);
	/**
	* \brief computes the component-wise value of (x -> sinh(x))
	*/
	UNARY_OP_ACCESSOR(cwiseSinh);
	/**
	* \brief computes the component-wise value of (x -> cosh(x))
	*/
	UNARY_OP_ACCESSOR(cwiseCosh);
	/**
	* \brief computes the component-wise value of (x -> tanh(x))
	*/
	UNARY_OP_ACCESSOR(cwiseTanh);
	/**
	* \brief computes the component-wise value of (x -> asinh(x))
	*/
	UNARY_OP_ACCESSOR(cwiseAsinh);
	/**
	* \brief computes the component-wise value of (x -> acosh(x))
	*/
	UNARY_OP_ACCESSOR(cwiseAcosh);
	/**
	* \brief computes the component-wise value of (x -> atanh(x))
	*/
	UNARY_OP_ACCESSOR(cwiseAtanh);
	/**
	* \brief Component-wise rounds up the entries to the next larger integer.
	* For an integer matrix, this does nothing
	*/
	UNARY_OP_ACCESSOR(cwiseCeil);
	/**
	* \brief Component-wise rounds down the entries to the next smaller integer.
	* For an integer matrix, this does nothing
	*/
	UNARY_OP_ACCESSOR(cwiseFloor);
	/**
	* \brief Component-wise rounds the entries to the next integer.
	* For an integer matrix, this does nothing
	*/
	UNARY_OP_ACCESSOR(cwiseRound);
	/**
	* \brief Calculate the error function of the input argument component-wise (x -> erf(x))
	*/
	UNARY_OP_ACCESSOR(cwiseErf);
	/**
	* \brief Calculate the complementary error function of the input argument component-wise (x -> erfc(x))
	*/
	UNARY_OP_ACCESSOR(cwiseErfc);
	/**
	* \brief Calculate the natural logarithm of the absolute value of the gamma function of the input argument component-wise (x -> lgamma(x))
	*/
	UNARY_OP_ACCESSOR(cwiseLgamma);

	/**
	* \brief Calculate the component-wise binary negation (x -> ~x).
	* Only available for integer matrices.
	*/
	UNARY_OP_ACCESSOR(cwiseBinaryNot);

	/**
	* \brief Calculate the component-wise logical negation (x -> !x).
	* Only available for boolean matrices.
	*/
	UNARY_OP_ACCESSOR(cwiseLogicalNot);

	/**
	* \brief Conjugates the matrix. This is a no-op for non-complex matrices.
	*/
	UNARY_OP_ACCESSOR(conjugate);

#undef UNARY_OP_ACCESSOR

	/**
	 * \brief Negates this matrix
	 */
	UnaryOp<_Derived, functor::UnaryMathFunctor_cwiseNegate<Scalar> > operator-() const {
		CUMAT_ERROR_IF_NO_NVCC(negate)
			return UnaryOp<_Derived, functor::UnaryMathFunctor_cwiseNegate <Scalar> >(derived());
	}

	/**
	 * \brief Custom unary expression.
	 * The unary functor must look as follow:
	 * \code
	 * struct MyFunctor
	 * {
	 *     typedef OutputType ReturnType;
	 *     __device__ CUMAT_STRONG_INLINE ReturnType operator()(const InputType& v, Index row, Index col, Index batch) const
	 *     {
	 *         return ...
	 *     }
	 * };
	 * \endcode
	 * with \c InputType being the type of this matrix expression and \c OutputType the output type.
	 */
	template<typename Functor>
	UnaryOp<_Derived, Functor> unaryExpr(const Functor& functor = Functor()) const
	{
		CUMAT_ERROR_IF_NO_NVCC(unaryExpr)
			return UnaryOp<_Derived, Functor>(derived(), functor);
	}

	/**
	 * \brief Transposes this matrix
	 */
	TransposeOp<_Derived, false> transpose() const
	{
		return TransposeOp<_Derived, false>(derived());
	}

	/**
	* \brief Returns the adjoint of this matrix (the conjugated transpose).
	*/
	TransposeOp<_Derived, true> adjoint() const
	{
		return TransposeOp<_Derived, true>(derived());
	}

	/**
	 * \brief Casts this matrix into a matrix of the target datatype
	 * \tparam _Target the target type
	 */
	template<typename _Target>
	CastingOp<_Derived, _Target> cast() const
	{
		CUMAT_ERROR_IF_NO_NVCC(cast)
			return CastingOp<_Derived, _Target>(derived());
	}

	/**
	 * \brief Returns a diagonal matrix with this vector as the main diagonal.
	 * This is only available for compile-time row- or column vectors.
	 */
	AsDiagonalOp<_Derived> asDiagonal() const
	{
		CUMAT_ERROR_IF_NO_NVCC(asDiagonal)
			return AsDiagonalOp<_Derived>(derived());
	}

	/**
	 * \brief Extracts the main diagonal of this matrix and returns it as a column vector.
	 * The matrix must not necessarily be square.
	 */
	ExtractDiagonalOp<_Derived> diagonal() const
	{
		CUMAT_ERROR_IF_NO_NVCC(diagonal)
			return ExtractDiagonalOp<_Derived>(derived());
	}

#ifdef CUMAT_PARSED_BY_DOXYGEN

	/**
	 * \brief Extracts the real part of the complex matrix.
	 * On real matrices, this is a no-op.
	 */
	ExtractComplexPartOp<_Derived, false, false> real() const
	{
		return ExtractComplexPartOp<_Derived, false, false>(derived());
	}

#else

	/**
	 * \brief Extracts the real part of the complex matrix.
	 * Specialization for real matrices: no-op.
	 */
	template<typename S = typename internal::traits<_Derived>::Scalar,
		typename = typename std::enable_if<!internal::NumTraits<S>::IsComplex>::type>
	_Derived real() const
	{
		return derived();
	}

	/**
	 * \brief Extracts the real part of the complex matrix.
	 * Specialization for complex matrices.
	 */
	template<typename S = typename internal::traits<_Derived>::Scalar,
		typename = typename std::enable_if<internal::NumTraits<S>::IsComplex>::type>
	ExtractComplexPartOp<_Derived, false, false> real() const
	{
		CUMAT_ERROR_IF_NO_NVCC(real)
			CUMAT_STATIC_ASSERT(internal::NumTraits<typename internal::traits<_Derived>::Scalar>::IsComplex, "Matrix must be complex");
		return ExtractComplexPartOp<_Derived, false, false>(derived());
	}

#endif

	/**
	 * \brief Extracts the imaginary part of the complex matrix.
	 * This method is only available for complex matrices.
	 */
	ExtractComplexPartOp<_Derived, true, false> imag() const
	{
		CUMAT_ERROR_IF_NO_NVCC(imag)
			CUMAT_STATIC_ASSERT(internal::NumTraits<typename internal::traits<_Derived>::Scalar>::IsComplex, "Matrix must be complex");
		return ExtractComplexPartOp<_Derived, true, false>(derived());
	}

	/**
	 * \brief Swaps the axis of this matrix.
	 *
	 * This operation is best explained on examples:
	 *  - <code>matrix.swapAxis<Column, Row, Batch>()</code>
	 *    returns the component-wise transpose
	 *  - <code>batchedVector.swapAxis<Row, Batch, NoAxis>()</code>
	 *    pulls in the batch dimension into the columns of the matrix.
	 *    The batch dimension is removed.
	 *  - <code>vector.swapAxis<NoAxis, NoAxis, Row>()</code>
	 *    converts the vector to a batched scalar tensor.
	 *
	 * \tparam _Row the axis which is used as the new row index
	 * \tparam _Col the axis which is used as the new column index
	 * \tparam _Batch the axis which is used as the new batch index
	 */
	template<Axis _Row, Axis _Col, Axis _Batch>
	SwapAxisOp<_Derived, _Row, _Col, _Batch> swapAxis() const
	{
		CUMAT_ERROR_IF_NO_NVCC(swapAxis)
			return SwapAxisOp<_Derived, _Row, _Col, _Batch>(derived());
	}

	//#include "BinaryOpsPlugin.inl"

#define BINARY_OP_ACCESSOR(Name) \
    template<typename _Right> \
	BinaryOp<_Derived, _Right, functor::BinaryMathFunctor_ ## Name <Scalar> > Name (const MatrixBase<_Right>& rhs) const { \
		CUMAT_ERROR_IF_NO_NVCC(Name)  \
		return BinaryOp<_Derived, _Right, functor::BinaryMathFunctor_ ## Name <Scalar> >(derived(), rhs.derived()); \
	} \
    template<typename _Right, typename T = typename std::enable_if<CUMAT_NAMESPACE internal::canBroadcast<_Right, Scalar>::value, \
        BinaryOp<_Derived, HostScalar<Scalar>, functor::BinaryMathFunctor_ ## Name <Scalar> > >::type > \
    T Name(const _Right& rhs) const { \
		CUMAT_ERROR_IF_NO_NVCC(Name)  \
		return BinaryOp<_Derived, HostScalar<Scalar>, functor::BinaryMathFunctor_ ## Name <Scalar> >(derived(), HostScalar<Scalar>(rhs)); \
	}
#define BINARY_OP_ACCESSOR_INV(Name) \
    template<typename _Left> \
        BinaryOp<_Left, _Derived, functor::BinaryMathFunctor_ ## Name <Scalar> > Name ## Inv(const MatrixBase<_Left>& lhs) const { \
		CUMAT_ERROR_IF_NO_NVCC(Name)  \
		return BinaryOp<_Left, _Derived, functor::BinaryMathFunctor_ ## Name <Scalar> >(lhs.derived(), derived()); \
	} \
    template<typename _Left, typename T = typename std::enable_if<CUMAT_NAMESPACE internal::canBroadcast<_Left, Scalar>::value, \
        BinaryOp<HostScalar<Scalar>, _Derived, functor::BinaryMathFunctor_ ## Name <Scalar> > >::type > \
    T Name ## Inv(const _Left& lhs) const { \
		CUMAT_ERROR_IF_NO_NVCC(Name)  \
		return BinaryOp<HostScalar<Scalar>, _Derived, functor::BinaryMathFunctor_ ## Name <Scalar> >(HostScalar<Scalar>(lhs), derived()); \
	}

/**
* \brief computes the component-wise multiplation (this*rhs)
*/
	BINARY_OP_ACCESSOR(cwiseMul)

		/**
		* \brief computes the component-wise dot-product.
		* The dot product of every individual element is computed,
		*  for regular matrices/vectors with scalar entries,
		*  this is exactly equal to the component-wise multiplication (\ref cwiseMul).
		* However, one can also use Matrix with submatrices/vectors as entries
		*  and then this operation might have the real dot-product
		*  if the respective functor \ref functor::BinaryMathFunctor_cwiseMDot is specialized.
		*/
		BINARY_OP_ACCESSOR(cwiseDot)

		/**
		* \brief computes the component-wise division (this/rhs)
		*/
		BINARY_OP_ACCESSOR(cwiseDiv)

		/**
		* \brief computes the inverted component-wise division (rhs/this)
		*/
		BINARY_OP_ACCESSOR_INV(cwiseDiv)

		/**
		* \brief computes the component-wise exponent (this^rhs)
		*/
		BINARY_OP_ACCESSOR(cwisePow)

		/**
		* \brief computes the inverted component-wise exponent (rhs^this)
		*/
		BINARY_OP_ACCESSOR_INV(cwisePow)

		/**
		* \brief computes the component-wise binary AND (this & rhs).
		* Only available for integer matrices.
		*/
		BINARY_OP_ACCESSOR(cwiseBinaryAnd)

		/**
		* \brief computes the component-wise binary OR (this | rhs).
		* Only available for integer matrices.
		*/
		BINARY_OP_ACCESSOR(cwiseBinaryOr)

		/**
		* \brief computes the component-wise binary XOR (this ^ rhs).
		* Only available for integer matrices.
		*/
		BINARY_OP_ACCESSOR(cwiseBinaryXor)

		/**
		* \brief computes the component-wise logical AND (this && rhs).
		* Only available for boolean matrices
		*/
		BINARY_OP_ACCESSOR(cwiseLogicalAnd)

		/**
		* \brief computes the component-wise logical OR (this || rhs).
		* Only available for boolean matrices
		*/
		BINARY_OP_ACCESSOR(cwiseLogicalOr)

		/**
		* \brief computes the component-wise logical XOR (this ^ rhs).
		* Only available for boolean matrices
		*/
		BINARY_OP_ACCESSOR(cwiseLogicalXor)

#undef BINARY_OP_ACCESSOR
#undef BINARY_OP_ACCESSOR_INV

		/**
		 * \brief Custom binary expression.
		 * The binary functor must support look as follow:
		 * \code
		 * struct MyFunctor
		 * {
		 *     typedef OutputType ReturnType;
		 *     __device__ CUMAT_STRONG_INLINE ReturnType operator()(const LeftType& x, const RightType& y, Index row, Index col, Index batch) const
		 *     {
		 *         return ...
		 *     }
		 * };
		 * \endcode
		 * where \c LeftType is the type of this matrix expression,
		 * \c RightType is the type of the rhs matrix,
		 * and \c OutputType is the output type.
		 *
		 * \param rhs the matrix expression on the right hand side
		 * \param functor the functor to apply component-wise
		 * \return an expression of a component-wise binary expression with a custom functor applied per component.
		 */
		template<typename Right, typename Functor>
	UnaryOp<_Derived, Functor> binaryExpr(const Right& rhs, const Functor& functor = Functor()) const
	{
		CUMAT_ERROR_IF_NO_NVCC(binaryExpr)
			return BinaryOp<_Derived, Right, Functor>(derived(), rhs.derived(), functor);
	}


	// #include "ReductionOpsPlugin.inl"

	/**
	 * \brief Computes the sum of all elements along the specified reduction axis
	 * \tparam axis the reduction axis, by default, reduction is performed among all axis
	 * \tparam Algorithm the reduction algorithm, a tag from the namespace ReductionAlg
	 */
	template<int axis = Axis::Row | Axis::Column | Axis::Batch, typename Algorithm = ReductionAlg::Auto>
	ReductionOp_StaticSwitched<_Derived, functor::Sum<Scalar>, axis, Algorithm> sum() const
	{
		CUMAT_ERROR_IF_NO_NVCC(sum)
			return ReductionOp_StaticSwitched<_Derived, functor::Sum<Scalar>, axis, Algorithm>(
				derived(), functor::Sum<Scalar>(), 0);
	}

	/**
	* \brief Computes the sum of all elements along the specified reduction axis
	* \param axis the reduction axis, by default, reduction is performed among all axis
	*/
	template<typename Algorithm = ReductionAlg::Auto>
	ReductionOp_DynamicSwitched<_Derived, functor::Sum<Scalar>, Algorithm> sum(int axis) const
	{
		CUMAT_ERROR_IF_NO_NVCC(sum)
			return ReductionOp_DynamicSwitched<_Derived, functor::Sum<Scalar>, Algorithm>(
				derived(), axis, functor::Sum<Scalar>(), 0);
	}

	/**
	* \brief Computes the product of all elements along the specified reduction axis
	* \tparam axis the reduction axis, by default, reduction is performed among all axis
	*/
	template<int axis = Axis::Row | Axis::Column | Axis::Batch, typename Algorithm = ReductionAlg::Auto>
	ReductionOp_StaticSwitched<_Derived, functor::Prod<Scalar>, axis, Algorithm> prod() const
	{
		CUMAT_ERROR_IF_NO_NVCC(prod)
			return ReductionOp_StaticSwitched<_Derived, functor::Prod<Scalar>, axis, Algorithm>(
				derived(), functor::Prod<Scalar>(), 1);
	}

	/**
	* \brief Computes the product of all elements along the specified reduction axis
	* \param axis the reduction axis, by default, reduction is performed among all axis
	*/
	template<typename Algorithm = ReductionAlg::Auto>
	ReductionOp_DynamicSwitched<_Derived, functor::Prod<Scalar>, Algorithm> prod(int axis) const
	{
		CUMAT_ERROR_IF_NO_NVCC(prod)
			return ReductionOp_DynamicSwitched<_Derived, functor::Prod<Scalar>, Algorithm>(
				derived(), axis, functor::Prod<Scalar>(), 1);
	}

	/**
	* \brief Computes the minimum value among all elements along the specified reduction axis
	* \tparam axis the reduction axis, by default, reduction is performed among all axis
	*/
	template<int axis = Axis::Row | Axis::Column | Axis::Batch, typename Algorithm = ReductionAlg::Auto>
	ReductionOp_StaticSwitched<_Derived, functor::Min<Scalar>, axis, Algorithm> minCoeff() const
	{
		CUMAT_ERROR_IF_NO_NVCC(minCoeff)
			return ReductionOp_StaticSwitched<_Derived, functor::Min<Scalar>, axis, Algorithm>(
				derived(), functor::Min<Scalar>(), std::numeric_limits<Scalar>::max());
	}

	/**
	* \brief Computes the minimum value among all elements along the specified reduction axis
	* \param axis the reduction axis, by default, reduction is performed among all axis
	*/
	template<typename Algorithm = ReductionAlg::Auto>
	ReductionOp_DynamicSwitched<_Derived, functor::Min<Scalar>, Algorithm> minCoeff(int axis) const
	{
		CUMAT_ERROR_IF_NO_NVCC(minCoeff)
			return ReductionOp_DynamicSwitched<_Derived, functor::Min<Scalar>, Algorithm>(
				derived(), axis, functor::Min<Scalar>(), std::numeric_limits<Scalar>::max());
	}

	/**
	* \brief Computes the maximum value among all elements along the specified reduction axis
	* \tparam axis the reduction axis, by default, reduction is performed among all axis
	*/
	template<int axis = Axis::Row | Axis::Column | Axis::Batch, typename Algorithm = ReductionAlg::Auto>
	ReductionOp_StaticSwitched<_Derived, functor::Max<Scalar>, axis, Algorithm> maxCoeff() const
	{
		CUMAT_ERROR_IF_NO_NVCC(maxCoeff)
			return ReductionOp_StaticSwitched<_Derived, functor::Max<Scalar>, axis, Algorithm>(
				derived(), functor::Max<Scalar>(), std::numeric_limits<Scalar>::lowest());
	}

	/**
	* \brief Computes the maximum value among all elements along the specified reduction axis
	* \param axis the reduction axis, by default, reduction is performed among all axis
	*/
	template<typename Algorithm = ReductionAlg::Auto>
	ReductionOp_DynamicSwitched<_Derived, functor::Max<Scalar>, Algorithm> maxCoeff(int axis) const
	{
		CUMAT_ERROR_IF_NO_NVCC(maxCoeff)
			return ReductionOp_DynamicSwitched<_Derived, functor::Max<Scalar>, Algorithm>(
				derived(), axis, functor::Max<Scalar>(), std::numeric_limits<Scalar>::lowest());
	}

	/**
	* \brief Computes the locical AND of all elements along the specified reduction axis,
	* i.e. <b>all</b> values must be true for the result to be true.
	* This is only defined for boolean matrices.
	* \tparam axis the reduction axis, by default, reduction is performed among all axis
	*/
	template<int axis = Axis::Row | Axis::Column | Axis::Batch, typename Algorithm = ReductionAlg::Auto>
	ReductionOp_StaticSwitched<_Derived, functor::LogicalAnd<Scalar>, axis, Algorithm> all() const
	{
		CUMAT_ERROR_IF_NO_NVCC(all)
			CUMAT_STATIC_ASSERT((std::is_same<Scalar, bool>::value), "'all' can only be applied to boolean matrices");
		return ReductionOp_StaticSwitched<_Derived, functor::LogicalAnd<Scalar>, axis, Algorithm>(
			derived(), functor::LogicalAnd<Scalar>(), true);
	}

	/**
	* \brief Computes the logical AND of all elements along the specified reduction axis,
	* i.e. <b>all</b> values must be true for the result to be true.
	* This is only defined for boolean matrices.
	* \param axis the reduction axis, by default, reduction is performed among all axis
	*/
	template<typename Algorithm = ReductionAlg::Auto>
	ReductionOp_DynamicSwitched<_Derived, functor::LogicalAnd<Scalar>, Algorithm> all(int axis) const
	{
		CUMAT_ERROR_IF_NO_NVCC(all)
			CUMAT_STATIC_ASSERT((std::is_same<Scalar, bool>::value), "'all' can only be applied to boolean matrices");
		return ReductionOp_DynamicSwitched<_Derived, functor::LogicalAnd<Scalar>, Algorithm>(
			derived(), axis, functor::LogicalAnd<Scalar>(), true);
	}

	/**
	* \brief Computes the locical OR of all elements along the specified reduction axis,
	* i.e. <b>any</b> value must be true for the result to be true.
	* This is only defined for boolean matrices.
	* \tparam axis the reduction axis, by default, reduction is performed among all axis
	*/
	template<int axis = Axis::Row | Axis::Column | Axis::Batch, typename Algorithm = ReductionAlg::Auto>
	ReductionOp_StaticSwitched<_Derived, functor::LogicalOr<Scalar>, axis, Algorithm> any() const
	{
		CUMAT_ERROR_IF_NO_NVCC(any)
			CUMAT_STATIC_ASSERT((std::is_same<Scalar, bool>::value), "'any' can only be applied to boolean matrices");
		return ReductionOp_StaticSwitched<_Derived, functor::LogicalOr<Scalar>, axis, Algorithm>(
			derived(), functor::LogicalOr<Scalar>(), false);
	}

	/**
	* \brief Computes the logical OR of all elements along the specified reduction axis,
	* i.e. <b>any</b> values must be true for the result to be true.
	* This is only defined for boolean matrices.
	* \param axis the reduction axis, by default, reduction is performed among all axis
	*/
	template<typename Algorithm = ReductionAlg::Auto>
	ReductionOp_DynamicSwitched<_Derived, functor::LogicalOr<Scalar>, Algorithm> any(int axis) const
	{
		CUMAT_ERROR_IF_NO_NVCC(any)
			CUMAT_STATIC_ASSERT((std::is_same<Scalar, bool>::value), "'any' can only be applied to boolean matrices");
		return ReductionOp_DynamicSwitched<_Derived, functor::LogicalOr<Scalar>, Algorithm>(
			derived(), axis, functor::LogicalOr<Scalar>(), false);
	}

	/**
	* \brief Computes the bitwise AND of all elements along the specified reduction axis.
	* This is only defined for matrices of integer types.
	* \tparam axis the reduction axis, by default, reduction is performed among all axis
	*/
	template<int axis = Axis::Row | Axis::Column | Axis::Batch, typename Algorithm = ReductionAlg::Auto>
	ReductionOp_StaticSwitched<_Derived, functor::BitwiseAnd<Scalar>, axis, Algorithm> bitwiseAnd() const
	{
		CUMAT_ERROR_IF_NO_NVCC(bitwiseAnd)
			CUMAT_STATIC_ASSERT(std::is_integral<Scalar>::value, "'bitwiseAnd' can only be applied to integral matrices");
		return ReductionOp_StaticSwitched<_Derived, functor::BitwiseAnd<Scalar>, axis, Algorithm>(
			derived(), functor::BitwiseAnd<Scalar>(), ~(Scalar(0)));
	}

	/**
	* \brief Computes the logical AND of all elements along the specified reduction axis.
	* This is only defined for matrices of integer types.
	* \param axis the reduction axis, by default, reduction is performed among all axis
	*/
	template<typename Algorithm = ReductionAlg::Auto>
	ReductionOp_DynamicSwitched<_Derived, functor::BitwiseAnd<Scalar>, Algorithm> bitwiseAnd(int axis) const
	{
		CUMAT_ERROR_IF_NO_NVCC(bitwiseAnd)
			CUMAT_STATIC_ASSERT(std::is_integral<Scalar>::value, "'bitwiseAnd' can only be applied to integral matrices");
		return ReductionOp_DynamicSwitched<_Derived, functor::BitwiseAnd<Scalar>, Algorithm>(
			derived(), axis, functor::BitwiseAnd<Scalar>(), ~(Scalar(0)));
	}

	/**
	* \brief Computes the bitwise OR of all elements along the specified reduction axis.
	* This is only defined for matrices of integer types.
	* \tparam axis the reduction axis, by default, reduction is performed among all axis
	*/
	template<int axis = Axis::Row | Axis::Column | Axis::Batch, typename Algorithm = ReductionAlg::Auto>
	ReductionOp_StaticSwitched<_Derived, functor::BitwiseOr<Scalar>, axis, Algorithm> bitwiseOr() const
	{
		CUMAT_ERROR_IF_NO_NVCC(bitwiseOr)
			CUMAT_STATIC_ASSERT(std::is_integral<Scalar>::value, "'bitwiseOr' can only be applied to integral matrices");
		return ReductionOp_StaticSwitched<_Derived, functor::BitwiseOr<Scalar>, axis, Algorithm>(
			derived(), functor::BitwiseOr<Scalar>(), Scalar(0));
	}

	/**
	* \brief Computes the logical OR of all elements along the specified reduction axis.
	* This is only defined for matrices of integer types.
	* \param axis the reduction axis, by default, reduction is performed among all axis
	*/
	template<typename Algorithm = ReductionAlg::Auto>
	ReductionOp_DynamicSwitched<_Derived, functor::BitwiseOr<Scalar>, Algorithm> bitwiseOr(int axis) const
	{
		CUMAT_ERROR_IF_NO_NVCC(bitwiseOr)
			CUMAT_STATIC_ASSERT(std::is_integral<Scalar>::value, "'bitwiseOr' can only be applied to integral matrices");
		return ReductionOp_DynamicSwitched<_Derived, functor::BitwiseOr<Scalar>, Algorithm>(
			derived(), axis, functor::BitwiseOr<Scalar>(), Scalar(0));
	}

	//custom reduction
	/**
	* \brief Custom reduction operation (static axis).
	* Here you can pass your own reduction operator and initial value.
	*
	* \param functor the reduction functor
	* \param initialValue the initial value to the reduction
	*
	* \tparam _Functor the reduction functor, must suppor the operation
	*   \code __device__ T operator()(const T &a, const T &b) const \endcode
	*   with \c T being the current scalar type
	* \tparam axis the reduction axis, by default, reduction is performed among all axis
	*/
	template<
		typename _Functor,
		int axis = Axis::Row | Axis::Column | Axis::Batch,
		typename Algorithm = ReductionAlg::Auto>
	ReductionOp_StaticSwitched<_Derived, _Functor, axis, Algorithm>
		reduction(const _Functor& functor = _Functor(), const Scalar& initialValue = Scalar(0)) const
	{
		CUMAT_ERROR_IF_NO_NVCC(reduction)
			return ReductionOp_StaticSwitched<_Derived, _Functor, axis, Algorithm>(
				derived(), functor, initialValue);
	}

	/**
	* \brief Custom reduction operation (dynamic axis).
	* Here you can pass your own reduction operator and initial value.
	*
	* \param axis the reduction axis, a combination of the constants in \ref Axis
	* \param functor the reduction functor
	* \param initialValue the initial value to the reduction
	*
	* \tparam _Functor the reduction functor, must support the operation
	*   \code __device__ T operator()(const T &a, const T &b) const \endcode
	*   with \c T being the current scalar type
	*/
	template<typename _Functor, typename Algorithm = ReductionAlg::Auto>
	ReductionOp_DynamicSwitched<_Derived, _Functor, Algorithm>
		reduction(int axis, const _Functor& functor = _Functor(), const Scalar& initialValue = Scalar(0)) const
	{
		CUMAT_ERROR_IF_NO_NVCC(reduction)
			return ReductionOp_DynamicSwitched<_Derived, _Functor, Algorithm>(
				derived(), axis, functor, initialValue);
	}

	//combined ops

	/**
	 * \brief Computes the trace of the matrix.
	 * This is simply implemented as <tt>*this.diagonal().sum<Axis::Column>()</tt>
	 */
	template<typename Algorithm = ReductionAlg::Auto>
	ReductionOp_StaticSwitched<
		ExtractDiagonalOp<_Derived>,
		functor::Sum<Scalar>,
		Axis::Row | Axis::Column,
		Algorithm> trace() const
	{
		CUMAT_ERROR_IF_NO_NVCC(trace)
			return diagonal().template sum<Axis::Row | Axis::Column, Algorithm>();
	}

	template<typename _Other, typename Algorithm = ReductionAlg::Auto>
	using DotReturnType = ReductionOp_StaticSwitched<
		BinaryOp<_Derived, _Other, functor::BinaryMathFunctor_cwiseDot<Scalar> >,
		functor::Sum<typename functor::BinaryMathFunctor_cwiseDot<Scalar>::ReturnType>,
		Axis::Row | Axis::Column,
		Algorithm>;
	/**
	 * \brief Computes the dot product between two vectors.
	 * This method is only allowed on compile-time vectors of the same orientation (either row- or column vector).
	 */
	template<typename _Other, typename Algorithm = ReductionAlg::Auto>
	DotReturnType<_Other, Algorithm> dot(const MatrixBase<_Other>& rhs) const
	{
		CUMAT_ERROR_IF_NO_NVCC(dot)
			CUMAT_STATIC_ASSERT(internal::traits<_Derived>::RowsAtCompileTime == 1 || internal::traits<_Derived>::ColsAtCompileTime == 1,
				"This matrix must be a compile-time row or column vector");
		CUMAT_STATIC_ASSERT(internal::traits<_Other>::RowsAtCompileTime == 1 || internal::traits<_Other>::ColsAtCompileTime == 1,
			"The right-hand-side must be a compile-time row or column vector");
		return ((*this).cwiseDot(rhs)).template sum<Axis::Row | Axis::Column, Algorithm>();
	}

	template<typename Algorithm = ReductionAlg::Auto>
	using SquaredNormReturnType = ReductionOp_StaticSwitched<
		UnaryOp<_Derived, functor::UnaryMathFunctor_cwiseAbs2<Scalar> >,
		functor::Sum<typename functor::UnaryMathFunctor_cwiseAbs2<Scalar>::ReturnType>,
		Axis::Row | Axis::Column,
		Algorithm>;
	/**
	 * \brief Computes the squared l2-norm of this matrix if it is a vecotr, or the squared Frobenius norm if it is a matrix.
	 * It consists in the the sum of the square of all the matrix entries.
	 */
	template<typename Algorithm = ReductionAlg::Auto>
	SquaredNormReturnType<Algorithm> squaredNorm() const
	{
		CUMAT_ERROR_IF_NO_NVCC(squaredNorm)
			return cwiseAbs2().template sum<Axis::Row | Axis::Column, Algorithm>();
	}

	template<typename Algorithm = ReductionAlg::Auto>
	using NormReturnType = UnaryOp<
		ReductionOp_StaticSwitched<
		UnaryOp<_Derived, functor::UnaryMathFunctor_cwiseAbs2<Scalar> >,
		functor::Sum<typename functor::UnaryMathFunctor_cwiseAbs2<Scalar>::ReturnType>,
		Axis::Row | Axis::Column,
		Algorithm>,
		functor::UnaryMathFunctor_cwiseSqrt<typename functor::UnaryMathFunctor_cwiseAbs2<Scalar>::ReturnType> >;
	/**
	 * \brief Computes the l2-norm of this matrix if it is a vecotr, or the Frobenius norm if it is a matrix.
	 * It consists in the square root of the sum of the square of all the matrix entries.
	 */
	template<typename Algorithm = ReductionAlg::Auto>
	NormReturnType<Algorithm> norm() const
	{
		CUMAT_ERROR_IF_NO_NVCC(norm)
			return squaredNorm<Algorithm>().cwiseSqrt();
	}

	//#include "DenseLinAlgPlugin.inl"

	/**
	 * \brief Computes and returns the LU decomposition with pivoting of this matrix.
	 * The resulting decomposition can then be used to compute the determinant of the matrix,
	 * invert the matrix and solve multiple linear equation systems.
	 */
	LUDecomposition<_Derived> decompositionLU() const
	{
		return LUDecomposition<_Derived>(derived());
	}

	/**
	 * \brief Computes and returns the Cholesky decompositionof this matrix.
	 * The matrix must be Hermetian and positive definite.
	 * The resulting decomposition can then be used to compute the determinant of the matrix,
	 * invert the matrix and solve multiple linear equation systems.
	 */
	CholeskyDecomposition<_Derived> decompositionCholesky() const
	{
		return CholeskyDecomposition<_Derived>(derived());
	}

	/**
	 * \brief Computes the determinant of this matrix.
	 * \return the determinant of this matrix
	 */
	DeterminantOp<_Derived> determinant() const
	{
		return DeterminantOp<_Derived>(derived());
	}

	/**
	* \brief Computes the log-determinant of this matrix.
	* This is only supported for hermitian positive definite matrices, because no sign is computed.
	* A negative determinant would return in an complex logarithm which requires to return
	* a complex result for real matrices. This is not desired.
	* \return the log-determinant of this matrix
	*/
	Matrix<typename internal::traits<_Derived>::Scalar, 1, 1, internal::traits<_Derived>::BatchesAtCompileTime, ColumnMajor> logDeterminant() const
	{
		//TODO: implement direct methods for matrices up to 4x4.
		return decompositionLU().logDeterminant();
	}

	/**
	 * \brief Computes the determinant of this matrix.
	 * For matrices of up to 4x4, an explicit formula is used. For larger matrices, this method falls back to a Cholesky Decomposition.
	 * \return the inverse of this matrix
	 */
	InverseOp<_Derived> inverse() const
	{
		return InverseOp<_Derived>(derived());
	}

	/**
	  * \brief Computation of matrix inverse and determinant in one kernel call.
	  *
	  * This is only for fixed-size square matrices of size up to 4x4.
	  *
	  * \param inverse Reference to the matrix in which to store the inverse.
	  * \param determinant Reference to the variable in which to store the determinant.
	  *
	  * \see inverse(), determinant()
	  */
	template<typename InverseType, typename DetType>
	void computeInverseAndDet(InverseType& inverseOut, DetType& detOut) const
	{
		CUMAT_STATIC_ASSERT(Rows >= 1 && Rows <= 4, "This matrix must be a compile-time 1x1, 2x2, 3x3 or 4x4 matrix");
		CUMAT_STATIC_ASSERT(Columns >= 1 && Columns <= 4, "This matrix must be a compile-time 1x1, 2x2, 3x3 or 4x4 matrix");
		CUMAT_STATIC_ASSERT(Rows == Columns, "This matrix must be symmetric");
		CUMAT_STATIC_ASSERT(Rows >= 1 && internal::traits<InverseType>::RowsAtCompileTime, "The output matrix must have the same compile-time size as this matrix");
		CUMAT_STATIC_ASSERT(Columns >= 1 && internal::traits<InverseType>::ColsAtCompileTime, "The output matrix must have the same compile-time size as this matrix");
		CUMAT_STATIC_ASSERT(CUMAT_IMPLIES(Batches > 0 && internal::traits<InverseType>::BatchesAtCompileTime > 0, Batches == internal::traits<InverseType>::BatchesAtCompileTime),
			"This matrix and the output matrix must have the same batch size");
		CUMAT_ASSERT_DIMENSION(batches() == inverseOut.batches());

		CUMAT_STATIC_ASSERT(internal::traits<DetType>::RowsAtCompileTime == 1, "The determinant output must be a (batched) scalar, i.e. compile-time 1x1 matrix");
		CUMAT_STATIC_ASSERT(internal::traits<DetType>::ColsAtCompileTime == 1, "The determinant output must be a (batched) scalar, i.e. compile-time 1x1 matrix");
		CUMAT_STATIC_ASSERT(CUMAT_IMPLIES(Batches > 0 && internal::traits<DetType>::BatchesAtCompileTime > 0, Batches == internal::traits<DetType>::BatchesAtCompileTime),
			"This matrix and the determinant matrix must have the same batch size");
		CUMAT_ASSERT_DIMENSION(batches() == detOut.batches());

		CUMAT_NAMESPACE ComputeInverseWithDet<_Derived, Rows, InverseType, DetType>::run(derived(), inverseOut, detOut);
	}

	//#include "SparseExpressionOpPlugin.inl"

	/**
	  * \brief Views this matrix expression as a sparse matrix.
	  * This enforces the specified sparsity pattern and the coefficients
	  * of this matrix expression are then only evaluated at these positions.
	  *
	  * For now, the only use case is the sparse matrix-vector product.
	  * For example:
	  * \code
	  * SparseMatrix<...> m1, m2; //all initialized with the same sparsity pattern
	  * VectorXf v1, v2 = ...;
	  * v1 = (m1 + m2)   * v2;
	  * \endcode
	  * In this form, this would trigger a dense matrix vector multiplication, which is
	  * completely unfeasable. This is because the the addition expression
	  * does not know anything about sparse matrices and the product operation
	  * then only sees an addition expression on the left hand side. Thus,
	  * because of lacking knowledge, it has to trigger a dense evaluation.
	  *
	  * Improvement:
	  * \code
	  * v1 = (m1 + m2).sparseView<Format>(m1.getSparsityPattern())   * v2;
	  * \endcode
	  * with Format being either CSR or CSC.
	  * This enforces the sparsity pattern of m1 onto the addition expression.
	  * Thus the (immediately following!) product expression sees this sparse expression
	  * and can trigger a sparse matrix-vector multiplication.
	  * But the sparse matrices \c m1 and \c m2 now have to search for their coefficients.
	  * This is because they don't know that the parent operations (add+multiply) will call
	  * their coefficients in order of their own entries. In other words, that the \c linear
	  * index parameter in \ref coeff(Index row, Index col, Index batch, Index linear) matches
	  * the linear index in their data array.
	  * (This is a valid assumption if you take transpose and block operations that change the
	  * access pattern into considerations. Further, this allows e.g. \c m2 to have a different
	  * sparsity pattern from m1, but only the entries that are included in both are used.)
	  *
	  * To overcome the above problem, one has to make one last adjustion:
	  * \code
	  * v1 = (m1.direct() + m2.direct()).sparseView<Format>(m1.getSparsityPattern())   * v2;
	  * \endcode
	  * \ref SparseMatrix::direct() tells the matrix that the linear index in
	  * \ref coeff(Index row, Index col, Index batch, Index linear) matches the linear index
	  * in the data array and thus can be used directly. This discards and checks that
	  * the row, column and batch index actually match. So use this with care
	  * if you know that access pattern is not changed in the operation.
	  * (This holds true for all non-broadcasting component wise expressions)
	  *
	  * \param pattern the enforced sparsity pattern
	  * \tparam _SparseFlags the sparse format: CSC or CSR
	  */
	template<SparseFlags _SparseFlags>
	SparseExpressionOp<Type, _SparseFlags>
		sparseView(const SparsityPattern<_SparseFlags>& pattern)
	{
		return SparseExpressionOp<Type, _SparseFlags>(derived(), pattern);
	}
};



template <typename _Derived, int _AccessFlags>
struct MatrixReadWrapper
{
private:
	enum
	{
		//the existing access flags
		flags = internal::traits<_Derived>::AccessFlags,
		//boolean if the access is sufficient
		sufficient = (flags & _AccessFlags)
	};
public:
	/**
	 * \brief The wrapped type: either the type itself, if the access is sufficient,
	 * or the evaluated type if not.
	 */
	using type = typename std::conditional<bool(sufficient), _Derived, typename _Derived::eval_t>::type;

	/*
	template<typename T = typename std::enable_if<sufficient, MatrixBase<_Derived>>::type>
	static type wrap(const T& m)
	{
		return m.derived();
	}
	template<typename T = typename std::enable_if<!sufficient, MatrixBase<_Derived>>::type>
	static type wrap(const T& m)
	{
		return m.derived().eval();
	}
	*/

private:
	MatrixReadWrapper() {} //not constructible
};

CUMAT_NAMESPACE_END

#endif