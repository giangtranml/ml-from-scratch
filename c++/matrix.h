//
// Created by Tran Giang on 7/25/22.
//

#ifndef ML_FROM_SCRATCH_MATRIX_H
#define ML_FROM_SCRATCH_MATRIX_H

template<typename T>
class Matrix{
private:
    int rows;
    int columns;
    T* elements;
public:
    Matrix(int rows, int columns);
};

#endif //ML_FROM_SCRATCH_MATRIX_H
