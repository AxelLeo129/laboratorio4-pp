/* File:     vector_add_modified.c
 *
 * Purpose:  Implement vector addition with random vector generation
 *
 * Compile:  gcc -g -Wall -o vector_add_modified vector_add_modified.c -lm
 * Run:      ./vector_add_modified
 *
 * Input:    The order of the vectors, n
 * Output:   The sum vector z = x+y and the first/last 10 elements of x, y, and z
 *
 * Note:
 *    If the program detects an error (order of vector <= 0 or malloc
 * failure), it prints a message and terminates
 *
 * IPP:      Section 3.4.6 (p. 109)
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void Read_n(int* n_p);
void Allocate_vectors(double** x_pp, double** y_pp, double** z_pp, int n);
void Print_vector(double b[], int n, char title[]);
void Vector_sum(double x[], double y[], double z[], int n);

/*---------------------------------------------------------------------*/
int main(void) {
   int n;
   double *x, *y, *z;
   clock_t start, end;
   double cpu_time_used;

   Read_n(&n);
   Allocate_vectors(&x, &y, &z, n);

   // Start measuring time
   start = clock();

   // Initialize x and y with random values
   srand(time(NULL));
   for (int i = 0; i < n; i++) {
      x[i] = (double)rand() / RAND_MAX;
      y[i] = (double)rand() / RAND_MAX;
   }


   Vector_sum(x, y, z, n);

   // Stop measuring time
   end = clock();
   cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

   // Print the time taken for the computation
   printf("Time taken for vector addition: %f seconds\n", cpu_time_used);

   free(x);
   free(y);
   free(z);

   return 0;
}  /* main */


/*---------------------------------------------------------------------
 * Function:  Read_n
 * Purpose:   Get the order of the vectors from stdin
 * Out arg:   n_p:  the order of the vectors
 *
 * Errors:    If n <= 0, the program terminates
 */
void Read_n(int* n_p /* out */) {
   printf("Enter the order of the vectors: ");
   scanf("%d", n_p);
   if (*n_p <= 0) {
      fprintf(stderr, "Order should be positive\n");
      exit(-1);
   }
}  /* Read_n */

/*---------------------------------------------------------------------
 * Function:  Allocate_vectors
 * Purpose:   Allocate storage for the vectors and initialize with random values
 * In arg:    n:  the order of the vectors
 * Out args:  x_pp, y_pp, z_pp:  pointers to storage for the vectors
 *
 * Errors:    If one of the mallocs fails, the program terminates
 */
void Allocate_vectors(
      double**  x_pp  /* out */, 
      double**  y_pp  /* out */, 
      double**  z_pp  /* out */, 
      int       n     /* in  */) {
   *x_pp = malloc(n * sizeof(double));
   *y_pp = malloc(n * sizeof(double));
   *z_pp = malloc(n * sizeof(double));
   if (*x_pp == NULL || *y_pp == NULL || *z_pp == NULL) {
      fprintf(stderr, "Can't allocate vectors\n");
      exit(-1);
   }

   // Initialize x and y with random values between 0 and 1
   srand(time(NULL));
   for (int i = 0; i < n; i++) {
      (*x_pp)[i] = (double)rand() / RAND_MAX;
      (*y_pp)[i] = (double)rand() / RAND_MAX;
   }
}  /* Allocate_vectors */

/*---------------------------------------------------------------------
 * Function:  Print_vector
 * Purpose:   Print the contents of a vector
 * In args:   b:  the vector to be printed
 *            n:  the order of the vector
 *            title:  title for print out
 */
void Print_vector(
      double  b[]     /* in */, 
      int     n       /* in */, 
      char    title[] /* in */) {
   int i;
   printf("%s\n", title);
   for (i = 0; i < n; i++)
      printf("%f ", b[i]);
   printf("\n");
}  /* Print_vector */

/*---------------------------------------------------------------------
 * Function:  Vector_sum
 * Purpose:   Add two vectors
 * In args:   x:  the first vector to be added
 *            y:  the second vector to be added
 *            n:  the order of the vectors
 * Out arg:   z:  the sum vector
 */
void Vector_sum(
      double  x[]  /* in  */, 
      double  y[]  /* in  */, 
      double  z[]  /* out */, 
      int     n    /* in  */) {
   for (int i = 0; i < n; i++)
      z[i] = x[i] + y[i];
}  /* Vector_sum */
