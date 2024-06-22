#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <omp.h>
#include <math.h>
#include <time.h>

int main() {
    
    double start, stop;
	clock_t cstart, cstop;

    int m = 2;
    int n = 100000000;
    int blockSize = 32768; 

    bool* result = (bool*)malloc((n - m + 1) * sizeof(bool));
    if (!result) {
        printf("Memory allocation failed for result array.\n");
        return 1;
    }
    memset(result, true, (n - m + 1) * sizeof(bool));

    bool* primeArray = (bool*)malloc((sqrt(n) + 1) * sizeof(bool));
    if (!primeArray) {
        printf("Memory allocation failed for primeArray.\n");
        free(result);
        return 1;
    }
    memset(primeArray, true, (sqrt(n) + 1) * sizeof(bool));

    start = omp_get_wtime();
	cstart = clock();

    for (int i = 2; i * i <= n; i++) {
        if (primeArray[i]) {
            for (int j = i * i; j <= n; j += i) {
                if (j <= sqrt(n) + 1) { 
                    primeArray[j] = false;
                }
            }
        }
    }

    int numberOfBlocks = (n - m) / blockSize;
    if ((n - m) % blockSize != 0) {
        numberOfBlocks++;
    }

    #pragma omp parallel for schedule (dynamic)

    for (int i = 0; i < numberOfBlocks; i++) {
        int low = m + i * blockSize;
        int high = low + blockSize - 1;
        if (high > n) {
            high = n;
        }
        for (int j = 2; j * j <= high; j++) {
            if (primeArray[j]) {
                int firstMultiple = (low / j) * j;
                if (firstMultiple < j * 2) {
                    firstMultiple = j * 2;
                }
                if (low % j) {
                    firstMultiple += j;
                }
                for (int k = firstMultiple; k <= high; k += j) {
                    if (k - m >= 0 && k - m < (n - m + 1)) {  
                        result[k - m] = false;
                    }
                }
            }
        }
    }

    cstop = clock();
	stop = omp_get_wtime();

    printf("Czas procesorów %f sekund \n", ((double)(cstop - cstart)/CLOCKS_PER_SEC));
	printf("Czas trwania obliczen - wallclock %f sekund \n", stop-start);

    free(primeArray);
    free(result);

    return 0;
}
