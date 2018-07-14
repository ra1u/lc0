#include <cstddef>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include "winograd_ispc.h"

static constexpr auto kWidth = 8;
static constexpr auto kHeight = 8;
static constexpr auto kSquares = kWidth * kHeight;

static constexpr auto kWtiles = (kWidth + 1) / 2; // 4
static constexpr auto kTiles = kWtiles * kWtiles; // 16

static constexpr auto kWinogradAlpha = 4;
static constexpr auto kWinogradTile = kWinogradAlpha * kWinogradAlpha;

void TransformIn(const size_t batch_size, const float *input,
                 const size_t channels, float *output)
{
    float x[kWinogradAlpha][kWinogradAlpha];
    float T1[kWinogradAlpha][kWinogradAlpha];

    for (size_t batch_index = 0; batch_index < batch_size; batch_index++) {
        const float *input_batch =
            input + batch_index * kWidth * kHeight * channels;
        float *V_batch = &output[channels * kTiles * batch_index];

        for (size_t channel = 0; channel < channels; channel++) {
            float *V_channel = V_batch + channel;
            const float *input_channel =
                input_batch + channel * (kWidth * kHeight);

            for (int block_y = 0; block_y < kWtiles; block_y++) {
                for (int block_x = 0; block_x < kWtiles; block_x++) {
                    // Tiles overlap by 2
                    const int yin = 2 * block_y - 1;
                    const int xin = 2 * block_x - 1;

                    for (int i = 0; i < kWinogradAlpha; i++) {
                        for (int j = 0; j < kWinogradAlpha; j++) {
                            if ((yin + i) >= 0 && (xin + j) >= 0 &&
                                (yin + i) < kHeight && (xin + j) < kWidth) {
                                x[i][j] = input_channel[(yin + i) * kWidth +
                                                        (xin + j)];
                            }
                            else {
                                x[i][j] = 0.0f;
                            }
                        }
                    }

                    // Calculates transpose(B).x.B
                    // B = [[ 1.0,  0.0,  0.0,  0.0],
                    //      [ 0.0,  1.0, -1.0,  1.0],
                    //      [-1.0,  1.0,  1.0,  0.0],
                    //      [ 0.0,  0.0,  0.0, -1.0]]

                    //     WinogradTile T1, T2;

                    T1[0][0] = x[0][0] - x[2][0];
                    T1[0][1] = x[0][1] - x[2][1];
                    T1[0][2] = x[0][2] - x[2][2];
                    T1[0][3] = x[0][3] - x[2][3];
                    T1[1][0] = x[1][0] + x[2][0];
                    T1[1][1] = x[1][1] + x[2][1];
                    T1[1][2] = x[1][2] + x[2][2];
                    T1[1][3] = x[1][3] + x[2][3];
                    T1[2][0] = x[2][0] - x[1][0];
                    T1[2][1] = x[2][1] - x[1][1];
                    T1[2][2] = x[2][2] - x[1][2];
                    T1[2][3] = x[2][3] - x[1][3];
                    T1[3][0] = x[1][0] - x[3][0];
                    T1[3][1] = x[1][1] - x[3][1];
                    T1[3][2] = x[1][2] - x[3][2];
                    T1[3][3] = x[1][3] - x[3][3];

                    const auto V_incr = channels * kTiles * batch_size;
                    float *wTile_V =
                        V_channel + channels * (block_y * kWtiles + block_x);

                    *wTile_V = T1[0][0] - T1[0][2];
                    wTile_V += V_incr;
                    *wTile_V = T1[0][1] + T1[0][2];
                    wTile_V += V_incr;
                    *wTile_V = T1[0][2] - T1[0][1];
                    wTile_V += V_incr;
                    *wTile_V = T1[0][1] - T1[0][3];
                    wTile_V += V_incr;
                    *wTile_V = T1[1][0] - T1[1][2];
                    wTile_V += V_incr;
                    *wTile_V = T1[1][1] + T1[1][2];
                    wTile_V += V_incr;
                    *wTile_V = T1[1][2] - T1[1][1];
                    wTile_V += V_incr;
                    *wTile_V = T1[1][1] - T1[1][3];
                    wTile_V += V_incr;
                    *wTile_V = T1[2][0] - T1[2][2];
                    wTile_V += V_incr;
                    *wTile_V = T1[2][1] + T1[2][2];
                    wTile_V += V_incr;
                    *wTile_V = T1[2][2] - T1[2][1];
                    wTile_V += V_incr;
                    *wTile_V = T1[2][1] - T1[2][3];
                    wTile_V += V_incr;
                    *wTile_V = T1[3][0] - T1[3][2];
                    wTile_V += V_incr;
                    *wTile_V = T1[3][1] + T1[3][2];
                    wTile_V += V_incr;
                    *wTile_V = T1[3][2] - T1[3][1];
                    wTile_V += V_incr;
                    *wTile_V = T1[3][1] - T1[3][3];
                }
            }
        }
    }
}

int main_naive(int argc, char *argv[])
{
    std::cout << "main_naive\n";
    const size_t bs_l = 256-30;
    const size_t bs_h = 256;
    const size_t channels = 192;
    std::vector<float> in(bs_h * kWidth * kHeight * kTiles * channels, 1.0);
    std::vector<float> out(channels * kTiles * kTiles * bs_h, 0);
    int N = 1;
    if (argc == 2) {
        N = std::stoi(argv[1]);
    }

    const auto start = std::chrono::high_resolution_clock::now();
    size_t c = 0;
    for (auto j = 0; j < N; ++j) {
        for (auto i = bs_l; i < bs_h; ++i) {
            TransformIn(i, &in[0], channels, &out[0]);
            c += i;
        }
    }
    const auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = stop - start;
    double perf = diff.count() / double(c);
    std::cout << " dummy " << out[0] << "\n";
    std::cout << "perf : total " << diff.count() 
              << "\n batches " << c
              << "\n perf " << perf << "\n";
}

int main_ispc(int argc, char *argv[])
{
    std::cout << "main_ispc\n";
    const uint32_t bs_l = 256-30;
    const uint32_t bs_h = 256;
    const uint32_t channels = 192;
    std::vector<float> in(bs_h * kWidth * kHeight * kTiles * channels, 1.0);
    std::vector<float> out(channels * kTiles * kTiles * bs_h, 0);
    int N = 1;
    if (argc == 2) {
        N = std::stoi(argv[1]);
    }

    const auto start = std::chrono::high_resolution_clock::now();
    size_t c = 0;
    for (auto j = 0; j < N; ++j) {
        for (auto i = bs_l; i < bs_h; ++i) {
            ispc::winograd_ispc(i, &in[0], channels, &out[0]);
            c += i;
        }
    }
    const auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = stop - start;
    double perf = diff.count() / double(c);
    std::cout << " dummy " << out[0] << "\n";
    std::cout << "perf : total " << diff.count() 
              << "\n batches " << c
              << "\n perf " << perf << "\n";
}


int main(int argc, char *argv[])
{
    return main_ispc(argc,argv);
}
