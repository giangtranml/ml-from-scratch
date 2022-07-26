#include <iostream>

template<typename T>
class Vector
{
private:
    int size;
    T* elements;
public:
    Vector(int size);

    ~Vector();

    T dot(Vector<T>& other);

    const T& operator[](int i) const;

    template<typename Scalar>
    Vector<T>& operator+(const Scalar& i);

    template<typename Scalar>
    Vector<T>& operator-(const Scalar& i);

    template<typename Scalar>
    Vector<T>& operator*(const Scalar& i);

    template<typename Scalar>
    Vector<T>& operator/(const Scalar& i);

    std::ostream& operator<<(std::ostream& outs);
};



