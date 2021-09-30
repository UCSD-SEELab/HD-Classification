#ifndef __CSV_H__
#define __CSV_H__

#include <string>
#include <vector>

class CSVReader
{
    std::string fileName;
    std::string delimeter;
 
public:
    CSVReader(std::string filename, std::string delm = ",") :
            fileName(filename), delimeter(delm)
    { }

    // Use this function to fetch data from a CSV File
    std::vector<std::vector<std::string> > getData();
};
 
#endif
