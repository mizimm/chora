#pragma once

#include <chrono>

namespace chora
{

std::chrono::time_point<std::chrono::high_resolution_clock> tic();
double toc(std::chrono::time_point<std::chrono::high_resolution_clock> starttime);	// get elapsed time in ms

}

#define TIC(x) std::chrono::time_point<std::chrono::high_resolution_clock> x = tic()
#define TOC(x) std::cout << #x << " " << toc(x) << " ms" << std::endl

