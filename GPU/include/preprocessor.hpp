#ifndef __PREPROCESSOR_H__
#define __PREPROCESSOR_H__

#include <vector>

void readChoirDat(char* filename, int& nFeatures, int& nClasses, std::vector<std::vector<float>>& X, std::vector<int>& y);

template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>> & vec) {   
    std::vector<T> result;
    for (const auto & v : vec)
        result.insert(result.end(), v.begin(), v.end());                                                                                         
    return result;
}

void l2norm(std::vector<std::vector<float>>& X);

#endif
