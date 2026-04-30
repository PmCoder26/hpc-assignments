#include <iostream>
#include <vector>
#include <fstream>
#include <omp.h>
#include <cstdlib>
using namespace std;

void printOpenMPDeviceInfo() {
    cout << "\n===== OpenMP Device Information =====\n";
#ifdef _OPENMP
    cout << "OpenMP Version Macro: " << _OPENMP << "\n";
#else
    cout << "OpenMP Version Macro: not available\n";
#endif
    cout << "Logical Processors: " << omp_get_num_procs() << "\n";
    cout << "Max Threads: " << omp_get_max_threads() << "\n";
    cout << "Thread Limit: " << omp_get_thread_limit() << "\n";
    cout << "Dynamic Threads Enabled: " << omp_get_dynamic() << "\n";
    cout << "Nested Parallelism Enabled: " << omp_get_nested() << "\n";
    cout << "Max Active Levels: " << omp_get_max_active_levels() << "\n";
    cout << "=====================================\n\n";
}

// ---------------- SEQUENTIAL BUBBLE SORT ----------------
void seqBubble(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
        }
    }
}

// ---------------- PARALLEL BUBBLE SORT ----------------
void parBubble(vector<int>& arr) {
    int n = arr.size();

    for (int i = 0; i < n; i++) {

        // Even phase
        #pragma omp parallel for
        for (int j = 0; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
        }

        // Odd phase			
        #pragma omp parallel for
        for (int j = 1; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
        }
    }
}

// ---------------- MERGE FUNCTION ----------------
void merge(vector<int>& arr, int l, int m, int r) {
    vector<int> temp(r - l + 1);

    int i = l, j = m + 1, k = 0;

    while (i <= m && j <= r) {
        if (arr[i] < arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }

    while (i <= m)
        temp[k++] = arr[i++];

    while (j <= r)
        temp[k++] = arr[j++];

    for (int i = 0; i < k; i++)
        arr[l + i] = temp[i];
}

// ---------------- SEQUENTIAL MERGE SORT ----------------
void seqMerge(vector<int>& arr, int l, int r) {
    if (l >= r) return;

    int m = (l + r) / 2;

    seqMerge(arr, l, m);
    seqMerge(arr, m + 1, r);

    merge(arr, l, m, r);
}

// ---------------- PARALLEL MERGE SORT ----------------
void parMerge(vector<int>& arr, int l, int r) {
    if (l >= r) return;

    int m = (l + r) / 2;

    #pragma omp parallel sections
    {
        #pragma omp section
        parMerge(arr, l, m);

        #pragma omp section
        parMerge(arr, m + 1, r);
    }

    merge(arr, l, m, r);
}

// ---------------- MAIN FUNCTION ----------------
int main() {

    printOpenMPDeviceInfo();

    ofstream file("sort_output.txt");
    file << "N,SEQ_TIME,PAR_TIME,SPEEDUP,EFFICIENCY\n";

    int sizes[] = {10, 100, 500, 10000, 20000};

    for (int N : sizes) {

        vector<int> arr(N);

        // Generate random input
        for (int i = 0; i < N; i++) {
            arr[i] = rand() % 1000;
        }

        vector<int> seqArr = arr;
        vector<int> parArr = arr;

        double start, end;

        // ---------------- SEQUENTIAL ----------------
        start = omp_get_wtime();

        seqBubble(seqArr);
        seqMerge(seqArr, 0, N - 1);

        end = omp_get_wtime();
        double seqTime = end - start;

        // ---------------- PARALLEL ----------------
        start = omp_get_wtime();

        parBubble(parArr);
        parMerge(parArr, 0, N - 1);

        end = omp_get_wtime();
        double parTime = end - start;

        // ---------------- PERFORMANCE ----------------
        double speedup = seqTime / parTime;
        int cores = omp_get_max_threads();
        double efficiency = speedup / cores;

        cout << "N = " << N
             << " | Seq = " << seqTime
             << " | Par = " << parTime
             << " | Speedup = " << speedup << endl;

        file << N << ","
             << seqTime << ","
             << parTime << ","
             << speedup << ","
             << efficiency << "\n";
    }

    file.close();

    cout << "\nResults stored in output.txt\n";

    return 0;
}

