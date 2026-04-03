#include<iostream>
#include<cstdlib>   // for rand()
#include<omp.h>     // OpenMP library for parallelism and timing
using namespace std;


// 🔹 Sorting class containing all algorithms
class Sorting {

    private:

        // 🔹 Utility function to swap two elements
        void swap(int arr[], int i, int j) {
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }

        // 🔹 Merge function used in Merge Sort
        // Combines two sorted subarrays into one sorted array
        void merge(int arr[], int start, int mid, int end) {

            int i = start;       // pointer for left subarray
            int j = mid + 1;     // pointer for right subarray
            int k = 0;           // pointer for temp array

            // Temporary array to store merged result
            int temp[end - start + 1];

            // Merge elements in sorted order
            while(i <= mid && j <= end) {
                if(arr[i] < arr[j])
                    temp[k++] = arr[i++];
                else
                    temp[k++] = arr[j++];
            }

            // Copy remaining elements of left subarray (if any)
            while(i <= mid)
                temp[k++] = arr[i++];

            // Copy remaining elements of right subarray (if any)
            while(j <= end)
                temp[k++] = arr[j++];

            // Copy merged elements back to original array
            for(k = 0, i = start; k < end - start + 1; k++, i++) {
                arr[i] = temp[k];
            }
        }

        // 🔹 Recursive helper for Parallel Merge Sort
        void parallelMergeSortUtil(int arr[], int start, int end) {

            // Base condition
            if(start >= end) return;

            int mid = (start + end) / 2;

            // Create task for left half
            #pragma omp task
            parallelMergeSortUtil(arr, start, mid);

            // Create task for right half
            #pragma omp task
            parallelMergeSortUtil(arr, mid + 1, end);

            // Wait for both tasks to finish
            #pragma omp taskwait

            // Merge sorted halves
            merge(arr, start, mid, end);
        }

    public:

        // 🔹 Utility function to print array
        void printArr(int arr[], int n) {
            for(int x = 0; x < n; x++) cout<<arr[x]<<" ";
            cout<<endl;
        }

        // ===================== BUBBLE SORT =====================

        // 🔹 Sequential Bubble Sort (O(n²))
        void sequentialBubbleSort(int arr[], int n) {
            for(int x = 0; x < n - 1; x++) {
                for(int y = 0; y < n - 1 - x; y++) {
                    if(arr[y] > arr[y + 1]) {
                        swap(arr, y, y + 1);
                    }
                }
            }
        }
        
        // 🔹 Parallel Bubble Sort (Odd-Even Transposition)
        void parallelBubbleSort(int arr[], int n) {

            // Repeat n phases
            for(int x = 0; x < n; x++) {

                bool sorted = true;  // used for early stopping

                // Even phase → compare (0,1), (2,3), ...
                if(x % 2 == 0) {
                    #pragma omp parallel for reduction(&&:sorted)
                    for(int y = 0; y < n - 1; y += 2) {
                        if(arr[y] > arr[y + 1]) {
                            swap(arr, y, y + 1);
                            sorted = false;
                        }
                    }
                } 
                // Odd phase → compare (1,2), (3,4), ...
                else {
                    #pragma omp parallel for reduction(&&:sorted)
                    for(int y = 1; y < n - 1; y += 2) {
                        if(arr[y] > arr[y + 1]) {
                            swap(arr, y, y + 1);
                            sorted = false;
                        }
                    }
                }

                // If no swaps happened → array already sorted
                if(sorted) break;
            }
        }        

        // ===================== MERGE SORT =====================

        // 🔹 Sequential Merge Sort (O(n log n))
        void sequentialMergeSort(int arr[], int start, int end) {
            if(start >= end) return;

            int mid = (start + end) / 2;

            sequentialMergeSort(arr, start, mid);
            sequentialMergeSort(arr, mid + 1, end);

            merge(arr, start, mid, end);
        }
        
        // 🔹 Parallel Merge Sort (uses OpenMP tasks)
        void parallelMergeSort(int arr[], int start, int end) {

            // Create parallel region
            #pragma omp parallel
            {
                // Only one thread initiates recursion
                #pragma omp single
                parallelMergeSortUtil(arr, start, end);
            }
        }
};


// ===================== MAIN FUNCTION =====================

int main() {

    Sorting s;

    int n = 100000;   // Large input size

    // 🔹 Allocate 4 arrays for fair comparison
    int *bubbleSeq = new int[n];
    int *bubblePar = new int[n];
    int *mergeSeq  = new int[n];
    int *mergePar  = new int[n];

    // 🔹 Generate random input data
    for(int i = 0; i < n; i++) {
        int val = rand() % 100000;
        bubbleSeq[i] = val;
        bubblePar[i] = val;
        mergeSeq[i]  = val;
        mergePar[i]  = val;
    }

    // ===================== BUBBLE SORT =====================

    cout << "========== BUBBLE SORT ==========\n";

    // Measure Sequential Bubble Sort time
    double start = omp_get_wtime();
    s.sequentialBubbleSort(bubbleSeq, n);
    double end = omp_get_wtime();
    double seqBubbleTime = end - start;

    cout << "Sequential Bubble Time: " << seqBubbleTime << " sec\n";

    // Measure Parallel Bubble Sort time
    start = omp_get_wtime();
    s.parallelBubbleSort(bubblePar, n);
    end = omp_get_wtime();
    double parBubbleTime = end - start;

    cout << "Parallel Bubble Time: " << parBubbleTime << " sec\n";

    // Compute speedup
    double speedup = seqBubbleTime / parBubbleTime;
    cout << "Speedup: " << speedup << endl;

    if(speedup > 1)
        cout << "Parallel is " << (speedup - 1) * 100 << "% faster\n";
    else
        cout << "Parallel is " << (1 - speedup) * 100 << "% slower\n";


    // ===================== MERGE SORT =====================

    cout << "\n========== MERGE SORT ==========\n";

    // Measure Sequential Merge Sort time
    start = omp_get_wtime();
    s.sequentialMergeSort(mergeSeq, 0, n - 1);
    end = omp_get_wtime();
    double seqMergeTime = end - start;

    cout << "Sequential Merge Time: " << seqMergeTime << " sec\n";

    // Measure Parallel Merge Sort time
    start = omp_get_wtime();
    s.parallelMergeSort(mergePar, 0, n - 1);
    end = omp_get_wtime();
    double parMergeTime = end - start;

    cout << "Parallel Merge Time: " << parMergeTime << " sec\n";

    // Compute speedup
    speedup = seqMergeTime / parMergeTime;
    cout << "Speedup: " << speedup << endl;

    if(speedup > 1)
        cout << "Parallel is " << (speedup - 1) * 100 << "% faster\n";
    else
        cout << "Parallel is " << (1 - speedup) * 100 << "% slower\n";

    // 🔹 Free dynamically allocated memory
    delete[] bubbleSeq;
    delete[] bubblePar;
    delete[] mergeSeq;
    delete[] mergePar;

    return 0;
}