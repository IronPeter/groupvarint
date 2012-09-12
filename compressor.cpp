#include <tmmintrin.h>
#include <emmintrin.h>
#include <vector>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

__m128i shuffles4[256];
__m128i shuffles2[256];
__m128i shuffle16 =  _mm_set_epi8(0xf, 0xe, 0xf, 0xe,
                                  0xf, 0xe, 0xf, 0xe,
                                  0xf, 0xe, 0xf, 0xe,
                                  0xf, 0xe, 0xf, 0xe);
size_t  sizes4[256];
size_t  sizes2[256];
typedef unsigned char ui8;
typedef unsigned int ui32;

void Init() {
    for (size_t i = 0; i < 256; ++i) {
        {
        ui8 buff[16];
        size_t k = 0;
        for (size_t j = 0; j < 4; ++j) {
            size_t size = (i >> (j * 2)) & 0x3;
            buff[j * 4 + 0] = size >= 0 ? k : 0xff;
            k += size >= 0;
            buff[j * 4 + 1] = size >= 1 ? k : 0xff;
            k += size >= 1;
            buff[j * 4 + 2] = size >= 2 ? k : 0xff;
            k += size >= 2;
            buff[j * 4 + 3] = size >= 3 ? k : 0xff;
            k += size >= 3;
        }
        sizes4[i] = k;
        shuffles4[i] = _mm_loadu_si128((__m128i *)buff);
        }
        {
        ui8 buff[16];
        size_t k = 0;
        for (size_t j = 0; j < 8; ++j) {
            size_t size = (i >> (j)) & 0x1;
            buff[j * 2 + 0] = size >= 0 ? k : 0xff;
            k += size >= 0;
            buff[j * 2 + 1] = size >= 1 ? k : 0xff;
            k += size >= 1;
        }
        sizes2[i] = k;
        shuffles2[i] = _mm_loadu_si128((__m128i *)buff);
        }
    }
}

__m128i Integrate4(__m128i v0, __m128i prev) {
    __m128i v1 = _mm_add_epi32(_mm_slli_si128(v0, 8), v0);
    __m128i v2 = _mm_add_epi32(_mm_slli_si128(v1, 4), v1);
    return _mm_add_epi32(v2, _mm_shuffle_epi32(prev, 0xff));
}

__m128i Integrate2(__m128i v0) {
    __m128i v1 = _mm_add_epi16(_mm_slli_si128(v0, 8), v0);
    __m128i v2 = _mm_add_epi16(_mm_slli_si128(v1, 4), v1);
    __m128i v3 = _mm_add_epi16(_mm_slli_si128(v2, 2), v2);
    return v3;
}

__m128i Integrate1(__m128i v0) {
    __m128i v1 = _mm_add_epi16(_mm_slli_si128(v0, 8), v0);
    __m128i v2 = _mm_add_epi16(_mm_slli_si128(v1, 4), v1);
    __m128i v3 = _mm_add_epi16(_mm_slli_si128(v2, 2), v2);
    __m128i v4 = _mm_add_epi16(_mm_slli_si128(v3, 1), v3);
    return v4;
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

ui8 *Code2(const ui32 *deltas, ui8 *code, ui8 *data) {
    ui8 t0 = Type(deltas[0]);
    ui8 t1 = Type(deltas[1]);
    ui8 t2 = Type(deltas[2]);
    ui8 t3 = Type(deltas[3]);
    ui8 t4 = Type(deltas[4]);
    ui8 t5 = Type(deltas[5]);
    ui8 t6 = Type(deltas[6]);
    ui8 t7 = Type(deltas[7]);
    code[0] = t0 + t1 * 2 + t2 * 4 + t3 * 8 + t4 * 16 + t5 * 32 + t6 * 64 + t7 * 128;
    ((int *)data)[0] = deltas[0];
    data += t0 + 1;
    ((int *)data)[0] = deltas[1];
    data += t1 + 1;
    ((int *)data)[0] = deltas[2];
    data += t2 + 1;
    ((int *)data)[0] = deltas[3];
    data += t3 + 1;
    ((int *)data)[0] = deltas[4];
    data += t4 + 1;
    ((int *)data)[0] = deltas[5];
    data += t5 + 1;
    ((int *)data)[0] = deltas[6];
    data += t6 + 1;
    ((int *)data)[0] = deltas[7];
    data += t7 + 1;
    return data;
}


ui8 *Code16(const ui32 *deltas, ui8 *code) {
    ui32 maxv = 0;
    for (size_t i = 0; i < 16; ++i) {
        maxv += deltas[i];
    }
    if (maxv < 0xff) {
        code[0] = 0;
        ++code;
        ui8 add = 0;
        for (size_t i = 0; i < 16; ++i) {
            add += (ui8)deltas[i];
            code[i] = add;
        }
        return code + 16;
    } else if (maxv < 0xffff) {
        code[0] = 1;
        ++code;
        ui8 *data = code + 2;
        data = Code2(deltas + 0 * 8, code + 0, data);
        data = Code2(deltas + 1 * 8, code + 1, data);
        return data;
    } else {
        code[0] = 2;
        ++code;
        ui8 *data = code + 4;
        data = Code4(deltas + 0 * 4, code + 0, data);
        data = Code4(deltas + 1 * 4, code + 1, data);
        data = Code4(deltas + 2 * 4, code + 2, data);
        data = Code4(deltas + 3 * 4, code + 3, data);
        return data;
    }
}


__attribute__((always_inline))
const ui8 *Decode16(const ui8 *src, volatile ui32 *dst, __m128i &last) {
    ui8 val = src[0];
    ++src;
    if (val == 0) {
        __m128i vali = _mm_loadu_si128((const __m128i *)src);
        //__m128i vali = Integrate1(val);
        __m128i shuf = _mm_shuffle_epi32(last, 0xff);
        __m128i v16_0 = _mm_unpacklo_epi8(vali, _mm_setzero_si128());
        __m128i v16_1 = _mm_unpackhi_epi8(vali, _mm_setzero_si128());
        __m128i v32_0 = _mm_unpacklo_epi16(v16_0, _mm_setzero_si128());
        __m128i v32_1 = _mm_unpackhi_epi16(v16_0, _mm_setzero_si128());
        __m128i v32_2 = _mm_unpacklo_epi16(v16_1, _mm_setzero_si128());
        __m128i v32_3 = _mm_unpackhi_epi16(v16_1, _mm_setzero_si128());
        __m128i out0 = _mm_add_epi32(shuf, v32_0);
        __m128i out1 = _mm_add_epi32(shuf, v32_1);
        __m128i out2 = _mm_add_epi32(shuf, v32_2);
        __m128i out3 = _mm_add_epi32(shuf, v32_3);
        last = out3;
        _mm_store_si128((__m128i *)dst + 0, out0);
        _mm_store_si128((__m128i *)dst + 1, out1);
        _mm_store_si128((__m128i *)dst + 2, out2);
        _mm_store_si128((__m128i *)dst + 3, out3);
        return src + 16;
    } else if (val == 1) {
        size_t ind0 = src[0];
        size_t ind1 = src[1];
        size_t siz0 = sizes2[ind0];
        size_t siz1 = sizes2[ind1];
        src += 2;
        __m128i v16_0 = Integrate2(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)src), shuffles2[ind0]));
        src += siz0;
        __m128i v16_1 = Integrate2(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)src), shuffles2[ind1]));
        v16_1 = _mm_add_epi16(v16_1, _mm_shuffle_epi8(v16_0, shuffle16));
        src += siz1;
        __m128i shuf = _mm_shuffle_epi32(last, 0xff);
        __m128i v32_0 = _mm_unpacklo_epi16(v16_0, _mm_setzero_si128());
        __m128i v32_1 = _mm_unpackhi_epi16(v16_0, _mm_setzero_si128());
        __m128i v32_2 = _mm_unpacklo_epi16(v16_1, _mm_setzero_si128());
        __m128i v32_3 = _mm_unpackhi_epi16(v16_1, _mm_setzero_si128());
        __m128i out0 = _mm_add_epi32(shuf, v32_0);
        __m128i out1 = _mm_add_epi32(shuf, v32_1);
        __m128i out2 = _mm_add_epi32(shuf, v32_2);
        __m128i out3 = _mm_add_epi32(shuf, v32_3);
        last = out3;
        _mm_store_si128((__m128i *)dst + 0, out0);
        _mm_store_si128((__m128i *)dst + 1, out1);
        _mm_store_si128((__m128i *)dst + 2, out2);
        _mm_store_si128((__m128i *)dst + 3, out3);
        return src;
    } else {
        size_t ind0 = src[0];
        size_t ind1 = src[1];
        size_t ind2 = src[2];
        size_t ind3 = src[3];
        size_t siz0 = sizes4[ind0];
        size_t siz1 = sizes4[ind1];
        size_t siz2 = sizes4[ind2];
        size_t siz3 = sizes4[ind3];
        src += 4;
        __m128i out0 = Integrate4(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)src), shuffles4[ind0]), last);
        src += siz0;
        __m128i out1 = Integrate4(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)src), shuffles4[ind1]), out0);
        src += siz1;
        __m128i out2 = Integrate4(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)src), shuffles4[ind2]), out1);
        src += siz2;
        __m128i out3 = Integrate4(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)src), shuffles4[ind3]), out2);
        src += siz3;
        last = out3;
        _mm_store_si128((__m128i *)dst + 0, out0);
        _mm_store_si128((__m128i *)dst + 1, out1);
        _mm_store_si128((__m128i *)dst + 2, out2);
        _mm_store_si128((__m128i *)dst + 3, out3);
        return src;
    }
}

#define SIZE (1024 * 1024 * 4)

ui8 buffer[SIZE * (8 + 64 + 1)];
ui32 outbuffer[SIZE * 16] __attribute__((aligned(0x10)));
ui32 delta[SIZE * 16];

int main(int argc, char *argv[]) {
    if (argc != 2) {
      printf("usage %s <bits per number>\n", argv[0]);
      return 1;
    }
    Init();
    int bits = atoi(argv[1]);
    for (size_t j = 0; j < SIZE * 16; ++j) {
        delta[j] = (rand() % (1 << bits));
    }
    ui8 *code = buffer;
    for (size_t j = 0; j < SIZE * 16; j += 16) {
        code = Code16(delta + j, code);
    }
    double gigs = 0.0;
    float cl1 = clock();
    while(gigs < 10.0) {
        const ui8 *ptr = buffer;
        ui32 *dst = outbuffer;
        __m128i last = _mm_setzero_si128();
        for (size_t i = 0; i < SIZE; i += 4) {
            ptr = Decode16(ptr, dst + 0 * 16, last);
            ptr = Decode16(ptr, dst + 1 * 16, last);
            ptr = Decode16(ptr, dst + 2 * 16, last);
            ptr = Decode16(ptr, dst + 3 * 16, last);
        }
        gigs += SIZE * 16.0 / 1000.0 / 1000.0 / 1000.0;
    }
    float cl2 = clock();
    float secs = (cl2 - cl1) / float(CLOCKS_PER_SEC);

    const ui8 *ptr = buffer;
    ui32 *dst = outbuffer;
    __m128i last = _mm_setzero_si128();
    for (size_t i = 0; i < SIZE; ++i) {
        ptr = Decode16(ptr, dst, last);
        dst += 16;
    }

    for (size_t j = 0; j < SIZE * 16; ++j) {
        ui32 d0 = outbuffer[j] - (j == 0 ? 0 : outbuffer[j - 1]);
        ui32 d1 = delta[j];
        if (d0 != d1) {
            printf("%d:%d:%d\n", d0, d1, (int)j);
        }
    }
    printf("%f seconds %f billions of ui32 per second\n", secs, gigs / secs);
    return 0;
};