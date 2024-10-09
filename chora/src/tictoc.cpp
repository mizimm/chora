#include "tictoc.h"

namespace chora
{

std::chrono::time_point<std::chrono::high_resolution_clock> tic()
{
        return std::chrono::high_resolution_clock::now();
}

double toc(std::chrono::time_point<std::chrono::high_resolution_clock> starttime)	// get elapsed time in ms
{
        return std::chrono::duration_cast<std::chrono::milliseconds>(tic()-starttime).count();
}

}
