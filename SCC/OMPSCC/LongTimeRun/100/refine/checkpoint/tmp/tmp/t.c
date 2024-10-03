#include <omp.h>
#include <stdio.h>

int main() {
    int current_score[] = {301};  // Set initial score
    int local_current_score = current_score[0];  // Initialize local variable

    printf("Before parallel loop, local score is %d\n", local_current_score);

    while (1) {
        #pragma omp parallel for reduction(max:local_current_score) schedule(dynamic)
        for (int paralli = 0; paralli < 10; paralli++) {
            int thread_id = omp_get_thread_num();
            int total_threads = omp_get_num_threads();

            local_current_score = current_score[0];  // Initialize local variable
            
            // Optional: Update local_current_score based on some logic here
            printf("%d (th) thread of %d: local score is %d\n", thread_id, total_threads, local_current_score);  
        }

        printf("After parallel loop, local score is %d\n", local_current_score);
        
        // You may want to break the loop at some point, e.g., based on a condition
        break;
    }

    return 0;
}

