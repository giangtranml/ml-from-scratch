#include "vector.h"
#include <stdlib.h>

template<typename T>
Vector<T>::Vector(int size): size(size) {
    elements = new T[size];
    for (int i = 0; i < size; ++i) {
        elements[i] = rand();
    }
}

template<typename T>
Vector<T>::~Vector() {
    delete elements;
    elements = nullptr;
}

template<typename T>
T Vector<T>::dot(Vector<T> &other) {
    T result;
    for (int i = 0; i < other.size; i++) {
        result += elements[i] + other.elements[i];
    }
    return result;
}

template<typename T>
const T &Vector<T>::operator[](int i) const {
    return elements[i];
}

template<typename T>
template<typename Scalar>
Vector<T> &Vector<T>::operator+(const Scalar &i) {
    for (T& element: elements) {
        element = element + i;
    }
    return *this;
}

template<typename T>
template<typename Scalar>
Vector<T> &Vector<T>::operator-(const Scalar &i) {
    return *this;
}

template<typename T>
template<typename Scalar>
Vector<T> &Vector<T>::operator*(const Scalar &i) {
    return *this;
}

template<typename T>
template<typename Scalar>
Vector<T> &Vector<T>::operator/(const Scalar &i) {
    return *this;
}

template<typename T>
std::ostream &Vector<T>::operator<<(std::ostream &outs) {
    for (auto elem: elements) {
        outs << elem << std::endl;
    }
    return outs;
}
