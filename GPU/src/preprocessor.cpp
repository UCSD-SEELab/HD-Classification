#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <utility>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "../include/preprocessor.hpp"


void readChoirDat(char* filename, int& nFeatures, int& nClasses,
            std::vector<std::vector<float>>& X,
            std::vector<int>& y
        ) {
    printf("%s\n", filename);


    FILE* fp = fopen(filename, "r");
    fread(&nFeatures, sizeof(int), 1, fp);
    fread(&nClasses, sizeof(int), 1, fp);

    float tmpfloat;

    while (true) {
        std::vector<float> newDP;
                for (int f=0; f < nFeatures; ++f) {
            if (feof(fp)) {
                break;
            }

            fread(&tmpfloat, sizeof(float), 1, fp);
            newDP.push_back(tmpfloat);
                }

        if (newDP.size() != nFeatures)
            break;

        int l;
        fread(&l, sizeof(int), 1, fp);
        X.push_back(newDP);
        y.push_back(l);
    }

    fclose(fp);
}

void l2norm(std::vector<std::vector<float>>& X) {
    float sq_sum_of_elems = 0;
    for(auto& xrow : X) {
        sq_sum_of_elems = 0;
        for(auto& n : xrow) {
            sq_sum_of_elems += n * n;
        }
        sq_sum_of_elems = sqrt(sq_sum_of_elems);
        for(auto& n : xrow) {
            n /= sq_sum_of_elems;
        }
    }       
}
