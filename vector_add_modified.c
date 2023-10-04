#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

void Check_for_error(int local_ok, char fname[], char message[], MPI_Comm comm);
void Read_n(int* n_p, int* local_n_p, int my_rank, int comm_sz, MPI_Comm comm);
void Allocate_vectors(double** local_x_pp, double** local_y_pp, double** local_z_pp, int local_n, MPI_Comm comm);
void Generate_random_vector(double local_a[], int local_n);
void Parallel_vector_sum(double local_x[], double local_y[], double local_z[], int local_n);
double Parallel_dot_product(double local_x[], double local_y[], int local_n, MPI_Comm comm);
void Parallel_scalar_mult(double scalar, double local_vec[], int local_n);

int main(void) {
    int n, local_n;
    int comm_sz, my_rank;
    double *local_x, *local_y, *local_z;
    double dot_product_result, scalar;
    MPI_Comm comm;
    double start_time, end_time;

    MPI_Init(NULL, NULL);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &comm_sz);
    MPI_Comm_rank(comm, &my_rank);

    // Inicializar el generador de números aleatorios una vez al inicio
    srand(time(NULL) + my_rank);  // Se añade my_rank para asegurar diferentes semillas

    Read_n(&n, &local_n, my_rank, comm_sz, comm);

#ifdef DEBUG
    printf("Proc %d > n = %d, local_n = %d\n", my_rank, n, local_n);
#endif

    Allocate_vectors(&local_x, &local_y, &local_z, local_n, comm);

    start_time = MPI_Wtime();
    Generate_random_vector(local_x, local_n);
    Generate_random_vector(local_y, local_n);

    Parallel_vector_sum(local_x, local_y, local_z, local_n);
    dot_product_result = Parallel_dot_product(local_x, local_y, local_n, comm);

    if (my_rank == 0) {
        printf("Enter the scalar value to multiply with the vectors: ");
        scanf("%lf", &scalar);
    }
    MPI_Bcast(&scalar, 1, MPI_DOUBLE, 0, comm);

    Parallel_scalar_mult(scalar, local_x, local_n);
    Parallel_scalar_mult(scalar, local_y, local_n);

    end_time = MPI_Wtime();

    if (my_rank == 0) {
        printf("Dot product result = %f\n", dot_product_result);
        printf("Time elapsed = %e seconds\n", end_time - start_time);
    }

    free(local_x);
    free(local_y);
    free(local_z);

    MPI_Finalize();

    return 0;
}

void Generate_random_vector(double local_a[], int local_n) {
    int local_i;

    srand(time(NULL));
    for (local_i = 0; local_i < local_n; local_i++) {
        local_a[local_i] = ((double)rand() / RAND_MAX);
    }
}

void Parallel_vector_sum(double local_x[], double local_y[], double local_z[], int local_n) {
    int local_i;

    for (local_i = 0; local_i < local_n; local_i++)
        local_z[local_i] = local_x[local_i] + local_y[local_i];
}

double Parallel_dot_product(double local_x[], double local_y[], int local_n, MPI_Comm comm) {
    int local_i;
    double local_dot_product = 0.0, global_dot_product;

    for (local_i = 0; local_i < local_n; local_i++)
        local_dot_product += local_x[local_i] * local_y[local_i];

    MPI_Allreduce(&local_dot_product, &global_dot_product, 1, MPI_DOUBLE, MPI_SUM, comm);

    return global_dot_product;
}

void Parallel_scalar_mult(double scalar, double local_vec[], int local_n) {
    int local_i;

    for (local_i = 0; local_i < local_n; local_i++)
        local_vec[local_i] *= scalar;
}

/*-------------------------------------------------------------------
 * Function:  Check_for_error
 * Purpose:   Check whether any process has found an error.  If so,
 *            print message and terminate all processes.  Otherwise,
 *            continue execution.
 * In args:   local_ok:  1 if calling process has found an error, 0
 *               otherwise
 *            fname:     name of function calling Check_for_error
 *            message:   message to print if there's an error
 *            comm:      communicator containing processes calling
 *                       Check_for_error:  should be MPI_COMM_WORLD.
 *
 * Note:
 *    The communicator containing the processes calling Check_for_error
 *    should be MPI_COMM_WORLD.
 */
void Check_for_error(int local_ok, char fname[], char message[], MPI_Comm comm) {
    int ok;

    MPI_Allreduce(&local_ok, &ok, 1, MPI_INT, MPI_MIN, comm);
    if (ok == 0) {
        int my_rank;
        MPI_Comm_rank(comm, &my_rank);
        if (my_rank == 0) {
            fprintf(stderr, "Proc %d > In %s, %s\n", my_rank, fname, message);
            fflush(stderr);
        }
        MPI_Finalize();
        exit(-1);
    }
}

/*-------------------------------------------------------------------
 * Function:  Read_n
 * Purpose:   Get the order of the vectors from stdin on proc 0 and
 *            broadcast to other processes.
 * In args:   my_rank:    process rank in communicator
 *            comm_sz:    number of processes in communicator
 *            comm:       communicator containing all the processes
 *                        calling Read_n
 * Out args:  n_p:        global value of n
 *            local_n_p:  local value of n = n/comm_sz
 *
 * Errors:    n should be positive and evenly divisible by comm_sz
 */
void Read_n(int* n_p, int* local_n_p, int my_rank, int comm_sz, MPI_Comm comm) {
    int local_ok = 1;
    char *fname = "Read_n";

    if (my_rank == 0) {
        printf("What's the order of the vectors?\n");
        scanf("%d", n_p);
    }
    MPI_Bcast(n_p, 1, MPI_INT, 0, comm);
    if (*n_p <= 0 || *n_p % comm_sz != 0) local_ok = 0;
    Check_for_error(local_ok, fname, "n should be > 0 and evenly divisible by comm_sz", comm);
    *local_n_p = *n_p / comm_sz;
}

/*-------------------------------------------------------------------
 * Function:  Allocate_vectors
 * Purpose:   Allocate storage for x, y, and z
 * In args:   local_n:  the size of the local vectors
 *            comm:     the communicator containing the calling processes
 * Out args:  local_x_pp, local_y_pp, local_z_pp:  pointers to memory
 *               blocks to be allocated for local vectors
 *
 * Errors:    One or more of the calls to malloc fails
 */
void Allocate_vectors(double** local_x_pp, double** local_y_pp, double** local_z_pp, int local_n, MPI_Comm comm) {
    int local_ok = 1;
    char* fname = "Allocate_vectors";

    *local_x_pp = malloc(local_n * sizeof(double));
    *local_y_pp = malloc(local_n * sizeof(double));
    *local_z_pp = malloc(local_n * sizeof(double));

    if (*local_x_pp == NULL || *local_y_pp == NULL || *local_z_pp == NULL) local_ok = 0;
    Check_for_error(local_ok, fname, "Can't allocate local vector(s)", comm);
}

/*-------------------------------------------------------------------
 * Function:   Read_vector
 * Purpose:    Read a vector from stdin on process 0 and distribute
 *             among the processes using a block distribution.
 * In args:    local_n:  size of local vectors
 *             n:        size of global vector
 *             vec_name: name of vector being read (e.g., "x")
 *             my_rank:  calling process' rank in comm
 *             comm:     communicator containing calling processes
 * Out arg:    local_a:  local vector read
 *
 * Errors:     if the malloc on process 0 for temporary storage
 *             fails the program terminates
 *
 * Note:
 *    This function assumes a block distribution and the order
 *   of the vector evenly divisible by comm_sz.
 */
void Read_vector(double local_a[], int local_n, int n, char vec_name[], int my_rank, MPI_Comm comm) {
    double* a = NULL;
    int i;
    int local_ok = 1;
    char* fname = "Read_vector";

    if (my_rank == 0) {
        a = malloc(n * sizeof(double));
        if (a == NULL) local_ok = 0;
        Check_for_error(local_ok, fname, "Can't allocate temporary vector", comm);
        printf("Enter the vector %s\n", vec_name);
        for (i = 0; i < n; i++)
            scanf("%lf", &a[i]);
        MPI_Scatter(a, local_n, MPI_DOUBLE, local_a, local_n, MPI_DOUBLE, 0, comm);
        free(a);
    } else {
        Check_for_error(local_ok, fname, "Can't allocate temporary vector", comm);
        MPI_Scatter(a, local_n, MPI_DOUBLE, local_a, local_n, MPI_DOUBLE, 0, comm);
    }
}