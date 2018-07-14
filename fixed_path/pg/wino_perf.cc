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
    return 0;
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
    return 0;
}

int main_sse(int argc, char *argv[])
{
    std::cout << "main_sse\n";
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
            TransformInSSs (i, &in[0], channels, &out[0]);
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
    return 0;
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
    TransformInSSs(bs_h, &in[0], channels, &out2[0]);
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


int main(int argc, char *argv[])
{
    return test(); // main_sse(argc,argv);
}


union sse_i128 {
   int32_t i4[4];
   __m128i i128;
};


static void TransformInSSs(const size_t batch_size, const float *input,
                 const size_t channels, float *output)
{

    const __m128i range4 = _mm_set_epi32(3,2,1,0);
    //__m128i _mm_load_si128 (__m128i const* mem_addr)
    
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
                    // 
                    //const __m128i block_x_sse = range4;
                    //const __m128i yin = _mm_set1_epi32 (2 * block_y - 1);
                    //const __m128i xin =_mm_set_epi32(-1,1,3,5); // 2 * block_x - 1;
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
                    /*
                    */
                }
            }
        }
    }
}
