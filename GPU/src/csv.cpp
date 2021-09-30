
#include <stdio.h>
#include <iostream>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <iterator>

#include "../include/csv.hpp"

std::vector<std::vector<std::string> > CSVReader::getData()
{
    std::ifstream file(fileName);
 
    std::vector<std::vector<std::string> > dataList;
 
    std::string line = "";
    while (getline(file, line))
    {
        std::vector<std::string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
        dataList.push_back(vec);
    }
    // Close the File
    file.close();
 
    return dataList;
}
