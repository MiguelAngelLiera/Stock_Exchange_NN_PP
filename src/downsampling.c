int CAT(TYPE, _downsampling_convolution)(const TYPE * const restrict input, const size_t N,
                                         const REAL_TYPE * const restrict filter, const size_t F,
                                         TYPE * const restrict output,
                                         const size_t step, MODE mode)
{
    /* This convolution performs efficient downsampling by computing every
     * step'th element of normal convolution (currently tested only for step=1
     * and step=2).
     */

    size_t i = step - 1, o = 0;

    if(mode == MODE_PERIODIZATION)
        return CAT(TYPE, _downsampling_convolution_periodization)(input, N, filter, F, output, step, 1);

    if (mode == MODE_SMOOTH && N < 2)
        mode = MODE_CONSTANT_EDGE;

    // left boundary overhang
    for(; i < F && i < N; i+=step, ++o){
        TYPE sum = 0;
        size_t j;
        for(j = 0; j <= i; ++j)
            sum += filter[j]*input[i-j];

        switch(mode) {
        case MODE_SYMMETRIC:
            while (j < F){
                size_t k;
                for(k = 0; k < N && j < F; ++j, ++k)
                    sum += filter[j]*input[k];
                for(k = 0; k < N && j < F; ++k, ++j)
                    sum += filter[j]*input[N-1-k];
            }
            break;
        case MODE_ZEROPAD:
        default:
            break;
        }
        output[o] = sum;
    }

    // center (if input equal or wider than filter: N >= F)
    for(; i < N; i+=step, ++o){
        TYPE sum = 0;
        size_t j;
        for(j = 0; j < F; ++j)
            sum += input[i-j]*filter[j];
        output[o] = sum;
    }

    // center (if filter is wider than input: F > N)
    for(; i < F; i+=step, ++o){
        TYPE sum = 0;
        size_t j = 0;

        switch(mode) {
        case MODE_SYMMETRIC:
            // Included from original: TODO: j < F-_offset
            /* Iterate over filter in reverse to process elements away from
             * data. This gives a known first input element to process (N-1)
             */
            while (i - j >= N){
                size_t k;
                for(k = 0; k < N && i-j >= N; ++j, ++k)
                    sum += filter[i-N-j]*input[N-1-k];
                for(k = 0; k < N && i-j >= N; ++j, ++k)
                    sum += filter[i-N-j]*input[k];
            }
            break;
        case MODE_ZEROPAD:
        default:
            j = i - N + 1;
            break;
        }

        for(; j <= i; ++j)
            sum += filter[j]*input[i-j];

        switch(mode) {
        case MODE_SYMMETRIC:
            while (j < F){
                size_t k;
                for(k = 0; k < N && j < F; ++j, ++k)
                    sum += filter[j]*input[k];
                for(k = 0; k < N && j < F; ++k, ++j)
                    sum += filter[j]*input[N-1-k];
            }
            break;
        case MODE_ZEROPAD:
        default:
            break;
        }
        output[o] = sum;
    }

    // right boundary overhang
    for(; i < N+F-1; i += step, ++o){
        TYPE sum = 0;
        size_t j = 0;
        switch(mode) {
        case MODE_SYMMETRIC:
            // Included from original: TODO: j < F-_offset
            while (i - j >= N){
                size_t k;
                for(k = 0; k < N && i-j >= N; ++j, ++k)
                    sum += filter[i-N-j]*input[N-1-k];
                for(k = 0; k < N && i-j >= N; ++j, ++k)
                    sum += filter[i-N-j]*input[k];
            }
            break;
        case MODE_ZEROPAD:
        default:
            j = i - N + 1;
            break;
        }
        for(; j < F; ++j)
            sum += filter[j]*input[i-j];
        output[o] = sum;
    }
    return 0;
}