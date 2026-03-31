#include <iostream>
#include <omp.h>

using namespace std;

/* =========================================================
   1. PARALLEL DIRECTIVE
   ---------------------------------------------------------
   - Creates multiple threads
   - Each thread executes the same block
   ========================================================= */
void ompParallel() {
    #pragma omp parallel
    {
        // Each thread prints its ID
        cout << "Thread: " << omp_get_thread_num() << endl;
    }
}

/* =========================================================
   2. PARALLEL FOR
   ---------------------------------------------------------
   - Divides loop iterations among threads
   - Each iteration executed once by some thread
   ========================================================= */
void ompParallelFor() {
    #pragma omp parallel for
    for(int x = 0; x < 5; x++) {
        cout << "x=" << x 
             << " thread=" << omp_get_thread_num() << endl;
    }
}

/* =========================================================
   3. REDUCTION
   ---------------------------------------------------------
   - Avoids race condition in shared variable
   - Each thread has private copy → combined at end
   ========================================================= */
void ompReduction() {
    int sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for(int x = 1; x <= 100; x++) {
        sum += x;
    }

    cout << "The sum from 1 to 100 is " << sum << endl;
}

/* =========================================================
   4. CRITICAL
   ---------------------------------------------------------
   - Only ONE thread executes block at a time
   - Used to avoid race condition in I/O or shared data
   ========================================================= */
void ompCritical() {
    #pragma omp parallel
    {
        #pragma omp critical
        {
            cout << "Thread " << omp_get_thread_num() << endl;
        }
    }
}

/* =========================================================
   5. BARRIER
   ---------------------------------------------------------
   - Synchronization point
   - All threads must reach barrier before continuing
   ========================================================= */
void ompBarrier() {
    #pragma omp parallel
    {
        cout << "Before barrier: " 
             << omp_get_thread_num() << endl;

        #pragma omp barrier   // wait for all threads

        cout << "After barrier: " 
             << omp_get_thread_num() << endl;
    }
}

/* =========================================================
   6. SECTIONS
   ---------------------------------------------------------
   - Different independent tasks executed in parallel
   - Each section executed ONCE by ONE thread
   ========================================================= */
void ompSections() {
    #pragma omp parallel sections
    {
        #pragma omp section
        cout << "Task 1" << endl;

        #pragma omp section
        cout << "Task 2" << endl;

        #pragma omp section
        cout << "Task 3" << endl;

        #pragma omp section
        cout << "Task 4" << endl;
    }
}

/* =========================================================
   Helper Function for Tasks
   ========================================================= */
void work(int id) {
    cout << "Task " << id 
         << " executed by thread "
         << omp_get_thread_num() << endl;
}

/* =========================================================
   7. TASK + TASKWAIT
   ---------------------------------------------------------
   - Tasks are created dynamically
   - taskwait ensures all tasks complete before proceeding
   ========================================================= */
void ompTaskWaits() {
    #pragma omp parallel
    {
        #pragma omp single   // only ONE thread creates tasks
        {
            #pragma omp task
            cout << "Task 1\n";

            #pragma omp task
            cout << "Task 2\n";

            #pragma omp taskwait   // wait for both tasks

            cout << "All tasks completed\n";
        }
    }
}

/* =========================================================
   8. NOWAIT
   ---------------------------------------------------------
   - Removes implicit barrier after loop
   - Threads continue execution immediately
   ========================================================= */
void ompNoWait() {
    #pragma omp parallel
    {
        #pragma omp for nowait
        for(int x = 0; x < 10; x++) {
            cout << "Loop with x=" << x << endl;
        }

        // Threads do NOT wait → may execute early
        cout << "Thread " << omp_get_thread_num()
             << " continues without waiting" << endl;
    }
}

/* =========================================================
   9. TASKLOOP
   ---------------------------------------------------------
   - Converts loop into multiple tasks
   - Tasks are scheduled dynamically
   - Requires 'single' to avoid duplicate creation
   ========================================================= */
void ompTaskLoop() {
    #pragma omp parallel
    {
        #pragma omp single   // only one thread creates tasks
        {
            #pragma omp taskloop
            for(int x = 0; x < 10; x++) {

                // Use critical to avoid mixed output
                #pragma omp critical
                {
                    cout << "x=" << x 
                         << " thread " 
                         << omp_get_thread_num() << endl;
                }
            }
        }
    }
}

/* =========================================================
   MAIN FUNCTION
   ========================================================= 
*/
int main() {

    // ompParallel();        // basic threading
    // ompParallelFor();     // loop parallelism
    // ompReduction();       // safe summation
    // ompCritical();        // mutual exclusion
    // ompBarrier();         // synchronization
    // ompSections();        // independent tasks
    // ompTaskWaits();       // dynamic tasks
    // ompNoWait();          // skip waiting
    // ompTaskLoop();           // dynamic loop tasks

    return 0;
}