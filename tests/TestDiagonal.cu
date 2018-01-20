#include <catch/catch.hpp>

#include <cuMat/Core>

#include "Utils.h"

using namespace cuMat;

TEST_CASE("extract diagonal", "[unary]")
{
    int data[1][3][4] {
        {
            {1, 2, 3, 4},
            {5, 6, 7, 8},
            {9, 10, 11, 12}
        }
    };
    int expected[1][3][1] {
        {
            {1},
            {6},
            {11}
        }
    };
    
    MatrixXiR mat = MatrixXiR::fromArray(data);
    VectorXi vec = mat.diagonal();

    assertMatrixEquality(expected, vec);
}

TEST_CASE("as diagonal matrix", "[unary]")
{
    int data1[1][3][1]{
        {
            { 1 },
            { 2 },
            { 3 }
        }
    };
    int data2[1][1][3] {
        {
            {1, 2, 3}
        }
    };
    int expected[1][3][3] {
        {
            {1, 0, 0},
            {0, 2, 0},
            {0, 0, 3}
        }
    };

    VectorXiR columnVector = VectorXiR::fromArray(data1);
    RowVectorXiR rowVector = RowVectorXiR::fromArray(data2);

    MatrixXi result1 = columnVector.asDiagonal();
    MatrixXi result2 = rowVector.asDiagonal();
    assertMatrixEquality(expected, result1);
    assertMatrixEquality(expected, result2);
}