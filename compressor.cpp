#include <tmmintrin.h>
#include <emmintrin.h>
#include <vector>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

__m128i shuffles[256];
size_t  sizes[256];
typedef unsigned char ui8;
typedef unsigned int ui32;

void Init() {
    for (size_t i = 0; i < 256; ++i) {
        ui8 buff[16];
        size_t k = 0;
        for (size_t j = 0; j < 4; ++j) {
            size_t size = (i >> (j * 2)) & 0x3;
            buff[j * 4 + 0] =  size >= 0 ? k : 0xff;
            k += size >= 0;
            buff[j * 4 + 1] =  size >= 1 ? k : 0xff;
            k += size >= 1;
            buff[j * 4 + 2] =  size >= 2 ? k : 0xff;
            k += size >= 2;
            buff[j * 4 + 3] =  size >= 3 ? k : 0xff;
            k += size >= 3;
        }
        sizes[i] = k;
        shuffles[i] = _mm_loadu_si128((__m128i *)buff);
    }
}

__m128i Integrate(__m128i v0, __m128i prev) {
    __m128i v1 = _mm_add_epi32(_mm_slli_si128(v0, 8), v0);
    __m128i v2 = _mm_add_epi32(_mm_slli_si128(v1, 4), v1);
    return _mm_add_epi32(v2, _mm_shuffle_epi32(prev, 0xff));
}

ui8 Type(ui32 i) {
    if (i < 0xff)
        return 0;
    if (i < 0xffff)
        return 1;
    if (i < 0xffffff)
        return 2;
    return 3;
}

ui8 *Code4(const ui32 *deltas, ui8 *code, ui8 *data) {
    ui8 t0 = Type(deltas[0]);
    ui8 t1 = Type(deltas[1]);
    ui8 t2 = Type(deltas[2]);
    ui8 t3 = Type(deltas[3]);
    code[0] = t0 + t1 * 4 + t2 * 16 + t3 * 64;
    ((int *)data)[0] = deltas[0];
    data += t0 + 1;
    ((int *)data)[0] = deltas[1];
    data += t1 + 1;
    ((int *)data)[0] = deltas[2];
    data += t2 + 1;
    ((int *)data)[0] = deltas[3];
    data += t3 + 1;
    return data;
}

ui8 *Code16(const ui32 *deltas, ui8 *code) {
    ui8 *data = code + 4;
    data = Code4(deltas + 0 * 4, code + 0, data);
    data = Code4(deltas + 1 * 4, code + 1, data);
    data = Code4(deltas + 2 * 4, code + 2, data);
    data = Code4(deltas + 3 * 4, code + 3, data);
    return data;
}

const ui8 *Decode16(const ui8 *src, ui32 *volatile dst, __m128i &last) {
    size_t ind0 = src[0];
    size_t ind1 = src[1];
    size_t ind2 = src[2];
    size_t ind3 = src[3];
    size_t siz0 = sizes[ind0];
    size_t siz1 = sizes[ind1];
    size_t siz2 = sizes[ind2];
    size_t siz3 = sizes[ind3];
    src += 4;
    __m128i out0 = Integrate(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)src), shuffles[ind0]), last);
    src += siz0;
    __m128i out1 = Integrate(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)src), shuffles[ind1]), out0);
    src += siz1;
    __m128i out2 = Integrate(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)src), shuffles[ind2]), out1);
    src += siz2;
    __m128i out3 = Integrate(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)src), shuffles[ind3]), out2);
    src += siz3;
    last = out3;
    _mm_storeu_si128((__m128i *)dst + 0, out0);
    _mm_storeu_si128((__m128i *)dst + 1, out1);
    _mm_storeu_si128((__m128i *)dst + 2, out2);
    _mm_storeu_si128((__m128i *)dst + 3, out3);
    return src;
}

#define SIZE (1024 * 1024)

ui8 buffer[SIZE * (8 + 64)];
ui32 outbuffer[SIZE * 16];
ui32 delta[SIZE * 16];

int main() {
    Init();
    for (size_t j = 0; j < SIZE * 16; ++j) {
        delta[j] = (rand() % 256) + (rand() & 256) * (rand() % 256);
    }
    ui8 *code = buffer;
    for (size_t j = 0; j < SIZE * 16; j += 16) {
        code = Code16(delta + j, code);
    }
    double gigs = 0.0;
    float cl1 = clock();
    while(gigs < 1.0) {
        const ui8 *ptr = buffer;
        ui32 *dst = outbuffer;
        __m128i last = _mm_setzero_si128();
        for (size_t i = 0; i < SIZE; ++i) {
            ptr = Decode16(ptr, dst, last);
            dst += 16;
        }
        gigs += SIZE * 16.0 / 1024.0 / 1024.0 / 1024.0;
    }
    float cl2 = clock();
    float secs = (cl2 - cl1) / float(CLOCKS_PER_SEC);
    for (size_t j = 1; j < SIZE * 16; ++j) {
        ui32 d0 = outbuffer[j] - outbuffer[j - 1];
        ui32 d1 = delta[j];
        if (d0 != d1) {
            printf("%d:%d\n", d0, d1);
        }
    }
    printf("%f %f\n", secs, gigs / secs);
    return 0;
};