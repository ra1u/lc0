#include <cstddef>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include "winograd_ispc.h"
#include <random>

#include <immintrin.h>

static void TransformInSSs(const size_t batch_size, const float *input,
                           const size_t channels, float *output);

void torig(const size_t batch_size, const size_t channels);

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
//
void TransformIn3(const size_t batch_size, const float *input,
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

/*
// input shape
[bat][ch][y][x][16]
// output shape
[16][bat][y][x][ch]
*/

#define RESTRICT __restrict__
template <typename F>
static void transpose(F *RESTRICT src, F *RESTRICT dest,
                      const size_t bat, const size_t chan)
{
    size_t pb, pc, pxy, pt;
    pb = 0;
    pc = 0;
    pxy = 0;
    pt = 0;
    const size_t kWH = kTiles;

    for (size_t b = 0; b < bat; ++b) {
        pb = chan * kWH * b;
        for (size_t c = 0; c < chan; ++c) {
            pc = c;
            for (size_t xy = 0; xy < kWH; ++xy) {
                pxy = chan * xy;
                for (size_t t = 0; t < kTiles; ++t) {
                    pt = pb + pc + pxy + bat * kWH * chan * t;
                    dest[pt] = *src;
                    ++src;
                }
            }
        }
    }
}
/*
// input shape
[bat][ch][y][x][16]
// output shape
[16][bat][y][x][ch]
*/




void TransformInTrans(const size_t batch_size, const float *input,
                 const size_t channels, float *output)
{  
    static std::vector<float> Vtmp(256 * kTiles * kTiles  * 256,0);
    float *wTile_V = &Vtmp[0];
    float x[kWinogradAlpha][kWinogradAlpha];
    float T1[kWinogradAlpha][kWinogradAlpha];

    for (size_t batch_index = 0; batch_index < batch_size; batch_index++) {
        const float *input_batch =
            input + batch_index * kWidth * kHeight * channels;
        float *V_batch = &Vtmp[channels * kTiles * batch_index];

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

                    const size_t V_incr = 1; //channels * kTiles * batch_size;
                    //float *wTile_V =
                    //    V_channel + channels * (block_y * kWtiles + block_x);

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
                    wTile_V += V_incr;
                }
            }
        }
    }
    //static void transpose(float * __restrict__  src, float * __restrict__  dest, const size_t bat, const size_t chan)
    transpose<float>(&Vtmp[0],output,batch_size,channels);
}

void TransformInTransCacheFull(const size_t batch_size, const float *input,
                 const size_t channels, float *output)
{  
    static std::vector<float> Vtmp(256 * kTiles * kTiles  * 256,0);
    float *wTile_V = &Vtmp[0];
    float x[kWinogradAlpha][kWinogradAlpha];
    float T1[kWinogradAlpha][kWinogradAlpha];

    for (size_t batch_index = 0; batch_index < batch_size; batch_index++) {
        const float *input_batch =
            input + batch_index * kWidth * kHeight * channels;
        float *V_batch = &Vtmp[channels * kTiles * batch_index];

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

                    const size_t V_incr = 1; //channels * kTiles * batch_size;
                    //float *wTile_V =
                    //    V_channel + channels * (block_y * kWtiles + block_x);

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
                    wTile_V += V_incr;
                }
            }
        }
    }
    //static void transpose(float * __restrict__  src, float * __restrict__  dest, const size_t bat, const size_t chan)
    //transpose<float>(&Vtmp[0],output,batch_size,channels);
    for(size_t i=0;i<batch_size * kTiles * kTiles * channels;++i){
	float v =  Vtmp[i];
	output[i] = v+v;
    }
}


//
void TransformInCacheWrong(const size_t batch_size, const float *input,
                 const size_t channels, float *output)
{
    float *wTile_V = output;
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

                    const auto V_incr = 1;
                    //float *wTile_V =
                    //    V_channel + channels * (block_y * kWtiles + block_x);

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
//
void TransformIn2(const size_t batch_size, const float *input,
                  const size_t channels, float *output)
{
    static const size_t Par = 16;
    float x[Par][kWinogradAlpha][kWinogradAlpha];
    float T1[Par][kWinogradAlpha][kWinogradAlpha];

    for (size_t batch_index = 0; batch_index < batch_size; batch_index++) {
        const float *input_batch =
            input + batch_index * kWidth * kHeight * channels;
        float *V_batch = &output[channels * kTiles * batch_index];
        const size_t channel_step = Par;
        for (size_t channel_long = 0; channel_long < channels;
            channel_long += channel_step) {

            for (int block_y = 0; block_y < kWtiles; block_y++) {
                for (int block_x = 0; block_x < kWtiles; block_x++) {
                    // Tiles overlap by 2
                    const int yin = 2 * block_y - 1;
                    const int xin = 2 * block_x - 1;

                    for (size_t ch = 0; ch < channel_step; ++ch) {
                        const size_t channel = channel_long + ch;
                        float *V_channel = V_batch + channel;
                        const float *input_channel =
                            input_batch + channel * (kWidth * kHeight);
                        for (int i = 0; i < kWinogradAlpha; i++) {
                            for (int j = 0; j < kWinogradAlpha; j++) {
                                if ((yin + i) >= 0 && (xin + j) >= 0 &&
                                    (yin + i) < kHeight && (xin + j) < kWidth) {
                                    x[ch][i][j] = input_channel[(yin + i) * kWidth +
                                                            (xin + j)];
                                }
                                else {
                                    x[ch][i][j] = 0.0f;
                                }
                            }
                        }

                        // Calculates transpose(B).x.B
                        // B = [[ 1.0,  0.0,  0.0,  0.0],
                        //      [ 0.0,  1.0, -1.0,  1.0],
                        //      [-1.0,  1.0,  1.0,  0.0],
                        //      [ 0.0,  0.0,  0.0, -1.0]]

                        //     WinogradTile T1, T2;

                        T1[ch][0][0] = x[ch][0][0] - x[ch][2][0];
                        T1[ch][0][1] = x[ch][0][1] - x[ch][2][1];
                        T1[ch][0][2] = x[ch][0][2] - x[ch][2][2];
                        T1[ch][0][3] = x[ch][0][3] - x[ch][2][3];
                        T1[ch][1][0] = x[ch][1][0] + x[ch][2][0];
                        T1[ch][1][1] = x[ch][1][1] + x[ch][2][1];
                        T1[ch][1][2] = x[ch][1][2] + x[ch][2][2];
                        T1[ch][1][3] = x[ch][1][3] + x[ch][2][3];
                        T1[ch][2][0] = x[ch][2][0] - x[ch][1][0];
                        T1[ch][2][1] = x[ch][2][1] - x[ch][1][1];
                        T1[ch][2][2] = x[ch][2][2] - x[ch][1][2];
                        T1[ch][2][3] = x[ch][2][3] - x[ch][1][3];
                        T1[ch][3][0] = x[ch][1][0] - x[ch][3][0];
                        T1[ch][3][1] = x[ch][1][1] - x[ch][3][1];
                        T1[ch][3][2] = x[ch][1][2] - x[ch][3][2];
                        T1[ch][3][3] = x[ch][1][3] - x[ch][3][3];
                    }
                    const auto V_incr = channels * kTiles * batch_size;
                    const size_t channel = channel_long;
                    float *V_channel = V_batch + channel;
                    float *wTile_V =
                        V_channel + channels * (block_y * kWtiles + block_x);

		    #define M(a0, a1, op, b1, b2)                                                  \
			do {                                                                       \
			    const size_t idx = channel_step;                                           \
			    for (size_t i = 0; i < idx; ++i) {                                     \
				wTile_V[i] = T1[i][a0][a1] op T1[i][b1][b2];                             \
			    };                                                                     \
			    wTile_V += V_incr;                                                     \
			} while (0)

                    M(0, 0, -, 0, 2);
                    M(0, 1, +, 0, 2);
                    M(0, 2, -, 0, 1);
                    M(0, 1, -, 0, 3);
                    M(1, 0, -, 1, 2);
                    M(1, 1, +, 1, 2);
                    M(1, 2, -, 1, 1);
                    M(1, 1, -, 1, 3);
                    M(2, 0, -, 2, 2);
                    M(2, 1, +, 2, 2);
                    M(2, 2, -, 2, 1);
                    M(2, 1, -, 2, 3);
                    M(3, 0, -, 3, 2);
                    M(3, 1, +, 3, 2);
                    M(3, 2, -, 3, 1);
                    M(3, 1, -, 3, 3);
		    #undef M
                }
            }
        }
    }
}

void TransformInCo(const size_t batch_size, const float *input,
                   const size_t channels, float *output)
{
    static const size_t Par = 16;
    float x[kWinogradAlpha][kWinogradAlpha];
    float T1[kWinogradAlpha][kWinogradAlpha];
    float R[16][Par];
    for (size_t batch_index = 0; batch_index < batch_size; batch_index++) {
       size_t channels_rem = channels;
       const float *input_batch =
            input + batch_index * kWidth * kHeight * channels;
        float *V_batch = &output[channels * kTiles * batch_index];
        for (size_t channel_long = 0; channel_long < channels;
             channel_long += Par) {
	    const size_t channel_step = std::min<size_t>(Par,channels_rem);
	    channels_rem -= channel_step;
            for (int block_y = 0; block_y < kWtiles; block_y++) {
                for (int block_x = 0; block_x < kWtiles; block_x++) {
                    // Tiles overlap by 2
                    const int yin = 2 * block_y - 1;
                    const int xin = 2 * block_x - 1;

                    for (size_t ch = 0; ch < channel_step; ++ch) {
                        const size_t channel = channel_long + ch;

                        const float *input_channel =
                            input_batch + channel * (kWidth * kHeight);
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
			
			R[0][ch] = T1[0][0] - T1[0][2];
                        R[1][ch] = T1[0][1] + T1[0][2];
                        R[2][ch] = T1[0][2] - T1[0][1];
                        R[3][ch] = T1[0][1] - T1[0][3];
                        R[4][ch] = T1[1][0] - T1[1][2];
                        R[5][ch] = T1[1][1] + T1[1][2];
                        R[6][ch] = T1[1][2] - T1[1][1];
                        R[7][ch] = T1[1][1] - T1[1][3];
                        R[8][ch] = T1[2][0] - T1[2][2];
                        R[9][ch] = T1[2][1] + T1[2][2];
                        R[10][ch] = T1[2][2] - T1[2][1];
                        R[11][ch] = T1[2][1] - T1[2][3];
                        R[12][ch] = T1[3][0] - T1[3][2];
                        R[13][ch] = T1[3][1] + T1[3][2];
                        R[14][ch] = T1[3][2] - T1[3][1];
                        R[15][ch] = T1[3][1] - T1[3][3];
                    }
                    const size_t channel = channel_long;
                    float *V_channel = V_batch + channel_long;
                    const auto V_incr = channels * kTiles * batch_size;
                    float *wTile_V =
                        V_channel + channels * (block_y * kWtiles + block_x);
                    for (size_t i = 0; i < 16; ++i) {
                        for (size_t ch = 0; ch < channel_step; ++ch) {
                            wTile_V[ch] = R[i][ch];
                        }
                        wTile_V += V_incr;
                    }
                }
            }
        }
    }
}

int test()
{
    //const uint32_t bs_l = 256 - 30;
    const uint32_t bs_h = 256;
    const uint32_t channels = 192;
    std::vector<float> in(bs_h * kWidth * kHeight * kTiles * channels, 1.0);
    
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);
    const size_t N = in.size();
    for(size_t i=0; i < N; ++i)
    {
	in[i] = dis(gen);
    }
  
    std::vector<float> out1(channels * kTiles * kTiles * bs_h, 0);
    TransformIn(bs_h, &in[0], channels, &out1[0]);
    std::vector<float> out2(out1.size(),0);
    //ispc::winograd_ispc(bs_h, &in[0], channels, &out2[0]);
    TransformInCo(bs_h, &in[0], channels, &out2[0]);
    //TransformInSSs(bs_h, &in[0], channels, &out2[0]);
    const size_t X = out1.size();
    size_t diff = 0;
    float total = 0;
    for(size_t i=0;i<X;++i){
	if(out1[i] != out2[i]){
	    ++diff;
	}
        total += out1[i];
    }
    std::cout << " diff " << diff << " total " << total <<  "\n";
    return diff;
}


//
 void TransformInSSs(const size_t batch_size, const float *input,
                 const size_t channels, float *output)
{

    const __m128i range4 = _mm_set_epi32(3,2,1,0);
    
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

		    __m128 x[kWinogradAlpha];

                    for (int i = 0; i < kWinogradAlpha; i++) {
			const __m128i j = range4;
			const __m128i yi = _mm_set1_epi32(yin + i);  // yin + i
                        const __m128i xj = _mm_add_epi32(_mm_set1_epi32(xin),j); // xin + j
                        const __m128i yipos = _mm_cmpgt_epi32(yi,_mm_set1_epi32(-1)); //(yin + i) >= 0
                        const __m128i xjpos = _mm_cmpgt_epi32(xj,_mm_set1_epi32(-1)); //(xin + j) >= 0
                        const __m128i yilow = _mm_cmplt_epi32(yi,_mm_set1_epi32(kHeight));  //(yin + i) < kHeight
                        const __m128i xjlow = _mm_cmplt_epi32(xj,_mm_set1_epi32(kWidth));  // (xin + j) < kWidth
                        const __m128i flag = _mm_and_si128(_mm_and_si128(yipos,xjpos), _mm_and_si128(yilow,xjlow)); // yipos && xjpos && yilow && xjlow
                        const __m128 ic = _mm_loadu_ps(&input_channel[(yin + i) * kWidth + xin]); // input_channel load
                        x[i] = _mm_blendv_ps(_mm_set1_ps(0.0),ic, _mm_castsi128_ps(flag)); // flag ? ic  : 0.0 
                    };

                    // Calculates transpose(B).x.B
                    // B = [[ 1.0,  0.0,  0.0,  0.0],
                    //      [ 0.0,  1.0, -1.0,  1.0],
                    //      [-1.0,  1.0,  1.0,  0.0],
                    //      [ 0.0,  0.0,  0.0, -1.0]]

                    //     WinogradTile T1, T2;
                    
		    __m128 T1_sse[kWinogradAlpha];

                    T1_sse[0] = _mm_sub_ps(x[0],x[2]);
                    T1_sse[1] = _mm_add_ps(x[1],x[2]);
                    T1_sse[2] = _mm_sub_ps(x[2],x[1]);
                    T1_sse[3] = _mm_sub_ps(x[1],x[3]);
                    
                    typedef float row_t[kWinogradAlpha];
                    row_t* T1  =  reinterpret_cast<row_t*>(&T1_sse[0]);
                    const auto V_incr = channels * kTiles * batch_size;
                    float *wTile_V = &output[channels * kTiles * batch_index] +
                                     channel +
                                     channels * (block_y * kWtiles + block_x);

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
//
 void TransformInSSsCo(const size_t batch_size, const float *input,
                 const size_t channels, float *output)
{

    const __m128i range4 = _mm_set_epi32(3,2,1,0);
    
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

		    __m128 x[kWinogradAlpha];

                    for (int i = 0; i < kWinogradAlpha; i++) {
			const __m128i j = range4;
			const __m128i yi = _mm_set1_epi32(yin + i);  // yin + i
                        const __m128i xj = _mm_add_epi32(_mm_set1_epi32(xin),j); // xin + j
                        const __m128i yipos = _mm_cmpgt_epi32(yi,_mm_set1_epi32(-1)); //(yin + i) >= 0
                        const __m128i xjpos = _mm_cmpgt_epi32(xj,_mm_set1_epi32(-1)); //(xin + j) >= 0
                        const __m128i yilow = _mm_cmplt_epi32(yi,_mm_set1_epi32(kHeight));  //(yin + i) < kHeight
                        const __m128i xjlow = _mm_cmplt_epi32(xj,_mm_set1_epi32(kWidth));  // (xin + j) < kWidth
                        const __m128i flag = _mm_and_si128(_mm_and_si128(yipos,xjpos), _mm_and_si128(yilow,xjlow)); // yipos && xjpos && yilow && xjlow
                        const __m128 ic = _mm_loadu_ps(&input_channel[(yin + i) * kWidth + xin]); // input_channel load
                        x[i] = _mm_blendv_ps(_mm_set1_ps(0.0),ic, _mm_castsi128_ps(flag)); // flag ? ic  : 0.0 
                    };

                    // Calculates transpose(B).x.B
                    // B = [[ 1.0,  0.0,  0.0,  0.0],
                    //      [ 0.0,  1.0, -1.0,  1.0],
                    //      [-1.0,  1.0,  1.0,  0.0],
                    //      [ 0.0,  0.0,  0.0, -1.0]]

                    //     WinogradTile T1, T2;
                    
		    __m128 T1_sse[kWinogradAlpha];

                    T1_sse[0] = _mm_sub_ps(x[0],x[2]);
                    T1_sse[1] = _mm_add_ps(x[1],x[2]);
                    T1_sse[2] = _mm_sub_ps(x[2],x[1]);
                    T1_sse[3] = _mm_sub_ps(x[1],x[3]);
                    
                    typedef float row_t[kWinogradAlpha];
                    row_t* T1  =  reinterpret_cast<row_t*>(&T1_sse[0]);
                    const auto V_incr = channels * kTiles * batch_size;
                    float *wTile_V = &output[channels * kTiles * batch_index] +
                                     channel +
                                     channels * (block_y * kWtiles + block_x);

		    
		    __m128 T2_sse[4];

                    const auto swizle = [](int a, int b, int c, int d) -> int {
			return (a << 6) | (b << 4) | (c << 2) | (d << 0);
                    }; 
		    for (size_t i = 0; i < 4; ++i)
                    {
                        __m128 a = _mm_shuffle_ps(T2_sse[i], T2_sse[i],swizle(0, 1, 2, 1));
			__m128 b = _mm_shuffle_ps(T2_sse[i], T2_sse[i],swizle(2, 2, 1, 3));
			//const __m128 sig_flag = _mm_set_ps(-1,1,-1,-1);
			T2_sse[i] = _mm_add_ps(a,b); //_mm_mul_ps(b,sig_flag));
                    }

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

//
template <typename T> 
static auto test_helper(T fun,std::string name)
{
    return [fun,name](int argc, char *argv[]) -> int {
        std::cout << name << "\n";
        const uint32_t bs_l = 256 - 30;
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
                fun(i, &in[0], channels, &out[0]);
                c += i;
            }
        }
        const auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = stop - start;
        double perf = diff.count() / double(c);
        std::cout << " dummy " << out[0] << "\n";
        std::cout << "perf : total " << diff.count() << "\n batches " << c
                  << "\n perf " << perf << "\n";
        return 0;
    };
}

//
static void TransformInSSs_wrong(const size_t batch_size, const float *input,
                                 const size_t channels, float *output)
{
    float *wTile_V = output;
    const __m128i range4 = _mm_set_epi32(3, 2, 1, 0);
    const float *input_channel = input;
    for (size_t batch_index = 0; batch_index < batch_size; batch_index++) {
        for (size_t channel = 0; channel < channels; channel++) {
            const float *input_channel =
                input + batch_index * kWidth * kHeight * channels +
                channel * (kWidth * kHeight);
            
            for (int block_y = 0; block_y < kWtiles; block_y++) {
                for (int block_x = 0; block_x < kWtiles; block_x++) {
                    // Tiles overlap by 2
                    const int yin = 2 * block_y - 1;
                    const int xin = 2 * block_x - 1;

                    __m128 x[kWinogradAlpha];

                    for (int i = 0; i < kWinogradAlpha; i++) {
                        const __m128i j = range4;
                        const __m128i yi = _mm_set1_epi32(yin + i); // yin + i
                        const __m128i xj =
                            _mm_add_epi32(_mm_set1_epi32(xin), j); // xin + j
                        const __m128i yipos = _mm_cmpgt_epi32(
                            yi, _mm_set1_epi32(-1)); //(yin + i) >= 0
                        const __m128i xjpos = _mm_cmpgt_epi32(
                            xj, _mm_set1_epi32(-1)); //(xin + j) >= 0
                        const __m128i yilow = _mm_cmplt_epi32(
                            yi, _mm_set1_epi32(kHeight)); //(yin + i) < kHeight
                        const __m128i xjlow = _mm_cmplt_epi32(
                            xj, _mm_set1_epi32(kWidth)); // (xin + j) < kWidth
                        const __m128i flag = _mm_and_si128(
                            _mm_and_si128(yipos, xjpos),
                            _mm_and_si128(
                                yilow,
                                xjlow)); // yipos && xjpos && yilow && xjlow
                        const __m128 ic = _mm_loadu_ps(
                            &input_channel[(yin + i) * kWidth +
                                           xin]); // input_channel load
                        x[i] = _mm_blendv_ps(
                            _mm_set1_ps(0.0), ic,
                            _mm_castsi128_ps(flag)); // flag ? ic  : 0.0
                    };
                    
                    // Calculates transpose(B).x.B
                    // B = [[ 1.0,  0.0,  0.0,  0.0],
                    //      [ 0.0,  1.0, -1.0,  1.0],
                    //      [-1.0,  1.0,  1.0,  0.0],
                    //      [ 0.0,  0.0,  0.0, -1.0]]

                    //     WinogradTile T1, T2;

                    __m128 T1_sse[kWinogradAlpha];

                    T1_sse[0] = _mm_sub_ps(x[0], x[2]);
                    T1_sse[1] = _mm_add_ps(x[1], x[2]);
                    T1_sse[2] = _mm_sub_ps(x[2], x[1]);
                    T1_sse[3] = _mm_sub_ps(x[1], x[3]);

                    const size_t V_incr = 1;
                    typedef float row_t[kWinogradAlpha];
                    row_t *T1 = reinterpret_cast<row_t *>(&T1_sse[0]);

                    for (auto i = 0; i < 4; ++i) {
                        _mm_storeu_ps(wTile_V, T1_sse[i]);
                        wTile_V += 4;
                    }
                }
            }
        }
    }
}

//

void TransformInDemo(const size_t batch_size, float *input,
                     const size_t channels, float *output)
{
    int tt = 1;
    for (size_t batch_index = 0; batch_index < batch_size; batch_index++) {
        float *input_batch = input + batch_index * kWidth * kHeight * channels;
        float *V_batch = &output[channels * kTiles * batch_index];

        for (size_t channel = 0; channel < channels; channel++) {
            float *V_channel = V_batch + channel;
            float *input_channel = input_batch + channel * (kWidth * kHeight);

            for (int block_y = 0; block_y < kWtiles; block_y++) {
                for (int block_x = 0; block_x < kWtiles; block_x++) {

                    const int yin = 2 * block_y - 1;
                    const int xin = 2 * block_x - 1;

                    for (int i = 0; i < kWinogradAlpha; i++) {
                        for (int j = 0; j < kWinogradAlpha; j++) {
                            if ((yin + i) >= 0 && (xin + j) >= 0 &&
                                (yin + i) < kHeight && (xin + j) < kWidth) {
                                input_channel[(yin + i) * kWidth + (xin + j)] =
                                    1;
                            }
                        }
                    }

                    const auto V_incr = channels * kTiles * batch_size;
                    float *wTile_V =
                        V_channel + channels * (block_y * kWtiles + block_x);

                    for (auto i = 0; i < 16; ++i) {
                        *wTile_V = tt;
                        tt += 1;
                        wTile_V += V_incr;
                    }
                }
            }
        }
    }
}

int testX(int argc, char *argv[])
{
    std::cout << "aa"
              << "\n";
    const uint32_t bs_h = 10;
    const uint32_t channels = 20;
    std::vector<float> in(bs_h * kWidth * kHeight * channels, 0);
    std::vector<float> out(channels * kTiles * kTiles * bs_h , 0);
    int N = 1;
    if (argc == 2) {
        N = std::stoi(argv[1]);
    }

    const auto start = std::chrono::high_resolution_clock::now();
    size_t c = 0;
    /*
    for (auto j = 0; j < N; ++j) {
        for (auto i = bs_l; i < bs_h; ++i) {
            TransformInDemo(i, &in[0], channels, &out[0]);
            c += i;
        }
    }
    */
    TransformInDemo(bs_h, &in[0], channels, &out[0]);
    const auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = stop - start;
    double perf = diff.count() / double(c);
    std::cout << " dummy " << out[0] << "\n";
    std::cout << "perf : total " << diff.count() << "\n batches " << c
              << "\n perf " << perf << "\n";
    size_t X = 64;
    std::vector<float>& v = in;
    for (size_t j = 0; j < v.size(); j += X) {
        for (size_t i = 0; i < X; ++i) {
            if (v[j + i])
                std::cout << '*';
            else
                std::cout << '_';
        }
        std::cout << '\n';
    }
    return 0;
}




/*
// optimal
[bat][ch][y][x][16]
// actual
[16][bat][y][x][ch]
*/



void torig(const size_t batch_size, const size_t channels)
{
    std::vector<int> Vtmp(batch_size * kTiles * kTiles * channels, 0);
    int num = 0;
    for (size_t batch_index = 0; batch_index < batch_size; batch_index++) {
        int *V_batch = &Vtmp[channels * kTiles * batch_index];

        for (size_t channel = 0; channel < channels; channel++) {
            int *V_channel = V_batch + channel;

            for (int block_y = 0; block_y < kWtiles; block_y++) {
                for (int block_x = 0; block_x < kWtiles; block_x++) {
                    // Tiles overlap by 2
                    const int yin = 2 * block_y - 1;
                    const int xin = 2 * block_x - 1;

                    const auto V_incr = channels * kTiles * batch_size;
                    int *wTile_V =
                        V_channel + channels * (block_y * kWtiles + block_x);

                    for (int i = 0; i < 16; ++i) {
                        *wTile_V = num++;
                        wTile_V += V_incr;
                    }
                }
            }
        }
    }
    const size_t N = 16;
    for (size_t i = 0; i < Vtmp.size(); i += N) {
        for (int j = 0; j < N; ++j) {
            std::cout << Vtmp[j + i] << ',';
        }
        std::cout << '\n';
    }
}

//
void tor_fix(const size_t batch_size, const size_t channels)
{
    std::vector<int> V1(batch_size * kTiles * kTiles * channels , 0);
    std::vector<int> V2(batch_size * kTiles * kTiles * channels , 0);
    int num = 0;
    for (size_t i = 0; i < V1.size(); i++) {
        V1[i] = num;
        ++num;
    }
    transpose(&V1[0], &V2[0], batch_size, channels);
    const size_t N = 16;
    for (size_t i = 0; i < V2.size(); i += N) {
        for (int j = 0; j < N; ++j) {
            std::cout << V2[j + i] << ',';
        }
        std::cout << '\n';
    }
}
//

void TransformInTransXX(const size_t batch_size, const float *input,
                 const size_t channels, float *output)
{  
    static std::vector<float> Vtmp(256 * kTiles * kTiles * 256,0);
    float *wTile_V = &Vtmp[0];
    float x[kWinogradAlpha][kWinogradAlpha];
    float T1[kWinogradAlpha][kWinogradAlpha];

    for (size_t batch_index = 0; batch_index < batch_size; batch_index++) {
        const float *input_batch =
            input + batch_index * kWidth * kHeight * channels;
        float *V_batch = &Vtmp[channels * kTiles * batch_index];

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

                    const size_t V_incr = 1; //channels * kTiles * batch_size;
                    //float *wTile_V =
                    //    V_channel + channels * (block_y * kWtiles + block_x);

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
    //static void transpose(float * __restrict__  src, float * __restrict__  dest, const size_t bat, const size_t chan)
    transpose(&Vtmp[0],output,batch_size,channels);
}


int main(int argc, char *argv[])
{
    
    //auto f = test_helper(TransformIn,"naive");
    //auto f = test_helper(TransformIn2,"TransformIn2");
    auto f = test_helper(TransformInCo,"TransformInCo");
    //auto f = test_helper(TransformIn3,"TransformIn3");
    //auto f = test_helper(TransformInTransCacheFull,"TransformInTransCacheFull");
    //auto f = test_helper(TransformInCacheWrong,"naive_wrong");
    //auto f =  test_helper(TransformInSSs,"TransformInSSs");
    //auto f =  test_helper(TransformInSSs_wrong,"TransformInSSs_wrong");
    //auto f =  test_helper(TransformInTrans,"TransformInTrans");
    //auto f = test_helper(ispc::winograd_ispc,"ispc");
    f(argc,argv);
    //testX(argc,argv);
    test();
    //tor_fix(10,20);
    //torig(10,20);
}
