#include<iostream>
#include<omp.h>
#include<vector>
#include<cstdlib>
using namespace std;


class Sorting {

    private:

        void merge(vector<int>& arr, int start, int mid, int end) {
            vector<int> temp;
            temp.resize(end - start + 1);

            int m = start, n = mid + 1, k = 0;

            while(m <= mid && n <= end) {
                if(arr[m] < arr[n]) temp[k++] = arr[m++];
                else temp[k++] = arr[n++];
            }

            // for remaining left part.
            while(m <= mid) temp[k++] = arr[m++];

            // for remaining right part.
            while(n <= end) temp[k++] = arr[n++];

            // final merge.        
            for(k = 0, m = start; k < temp.size() && m <= end; k++, m++) {
                arr[m] = temp[k];
            }
        }

        void parallelMergeSort(vector<int>& arr, int start, int end) {
            if(start >= end) return;

            // IMPORTANT OPTIMIZATION
            if(end - start < 1000) {
                sequentialMergeSort(arr, start, end);
                return;
            }

            int mid = start + (end - start) / 2;
            #pragma omp task shared(arr)
            parallelMergeSort(arr, start, mid);

            #pragma omp task shared(arr)
            parallelMergeSort(arr, mid + 1, end);

            #pragma omp taskwait
            merge(arr, start, mid, end);
        }
        
    
    public:

    void sequentialBubble(vector<int>& arr) {

        for (int x = 0; x < arr.size() - 1; x++) {
            
            bool sorted = true;

            for (int y = 0; y < arr.size() - 1 - x; y++) {
                
                if(arr[y] > arr[y + 1]) {
                    sorted = false;
                    swap(arr[y], arr[y + 1]);                    
                }
            }

            if(sorted) break;
        }
    }

    void parallelBubble(vector<int>& arr) {

        int n = arr.size();
        for(int x = 0; x < n; x++) {

            // Even phase.
            #pragma omp parallel for
            for(int y = 0; y < n - 1; y += 2) {

                if(arr[y] > arr[y + 1]) {
                    swap(arr[y], arr[y + 1]);                  
                }
            }

            // Odd phase.
            #pragma omp parallel for
            for(int y = 1; y < n - 1; y += 2) {

                if(arr[y] > arr[y + 1]) {
                    swap(arr[y], arr[y + 1]);
                }
            }
        }
    }

    void sequentialMergeSort(vector<int>& arr, int start, int end) {
        if(start >= end) return;

        int mid = start + (end - start) / 2;

        sequentialMergeSort(arr, start, mid);
        sequentialMergeSort(arr, mid + 1, end);

        merge(arr, start, mid, end);
    }

    void parallelMergeSortStart(vector<int>& arr) {
        #pragma omp parallel
        {
            #pragma omp single 
            {
                parallelMergeSort(arr, 0, arr.size() - 1);
            }
        }
    }


};


int main() {

    int N = 1000000;
    vector<int> sequentiallArr;
    vector<int> parallelArr;    

    sequentiallArr.resize(N);
    parallelArr.resize(N);

    Sorting s;

    for(int x = 0; x < N; x++) {
        sequentiallArr[x] = rand() % 1000;
        parallelArr[x] = sequentiallArr[x];
    }

    double start, end, sequentialTime, parallelTime, speedUp;


    // start = omp_get_wtime();
    // s.sequentialBubble(sequentiallArr);
    // end = omp_get_wtime();
    // sequentialTime = end - start;

    // start = omp_get_wtime();
    // s.parallelBubble(parallelArr);
    // end = omp_get_wtime();
    // parallelTime = end - start;

    // speedUp = sequentialTime / parallelTime;

    // cout<<"************* Bubble Sort *************"<<endl;
    // cout<<"Type=SEQ, Time="<<sequentialTime<<endl;
    // cout<<"Type=PAR, Time="<<parallelTime<<endl;
    // cout<<"Speedup="<<speedUp<<endl;


    start = omp_get_wtime();
    s.sequentialMergeSort(sequentiallArr, 0, sequentiallArr.size() - 1);
    end = omp_get_wtime();
    sequentialTime = end - start;

    start = omp_get_wtime();
    s.parallelMergeSortStart(parallelArr);
    end = omp_get_wtime();
    parallelTime = end - start;

    speedUp = sequentialTime / parallelTime;

    cout<<"************* Merge Sort *************"<<endl;
    cout<<"Type=SEQ, Time="<<sequentialTime<<endl;
    cout<<"Type=PAR, Time="<<parallelTime<<endl;
    cout<<"Speedup="<<speedUp<<endl;


    


}