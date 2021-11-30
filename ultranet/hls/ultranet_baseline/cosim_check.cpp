#include <iostream>
#include <fstream>

int main(int argc, char** argv) {
    std::ifstream presim("ultranet_cosim/presim_out.dat");
    std::ifstream postsim("ultranet_cosim/postsim_out.dat");
    std::string presim_line, postsim_line;
    int line_no = 1;
    while (1) {
        if (std::getline(presim, presim_line)) {
            if (std::getline(postsim, postsim_line)) {
                if (presim_line != postsim_line) {
                    std::cout << "Mismatch at line " << line_no << std::endl;
                    std::cout << "  Expected: " << presim_line << std::endl;
                    std::cout << "  Actual  : " << postsim_line << std::endl;
                }
                line_no++;
            } else {
                std::cout << "Error reading postsim output data! Aborting..." << std::endl;
                break;
            }
        } else {
            break;
        }
    }
    presim.close();
    postsim.close();
    return 0;
}