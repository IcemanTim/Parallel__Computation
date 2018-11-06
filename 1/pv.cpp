#include <iostream>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include <omp.h>
#include <ctime>
#include <cstdlib>

using namespace std;

double Max(double x, double y){
    if (x < y)
        return y;
    else 
        return x;
}

/*----------------------------------------------------------*/
//Создание матрицы с элементами dii = 1/aii

void makediag(double *DD, double *Aval, int *AI, int *AJ, int N) {

    int num = AI[0];
    int row = 0;
    
    int i;
    for(int i = 1; i < N+1; i++){
        for(int j = num; j < AI[i]; j++){
            if(row == AJ[j]){
                DD[row] = 1.0 / Aval[j]; 
                break;
            }
        }
        row++;
        num = AI[i];
    }
    return;
} 

/*----------------------------------------------------------*/
//Копирование матрицы

void repeatmatr(double *X, double *Y, int N) {

    int i;
    #pragma omp parallel for private(i)
        for (i = 0; i < N; i++)
            X[i] = Y[i];
    return;
}

/*----------------------------------------------------------*/
//SpMV : SpMV(DD,PP,PP2); SpMV(A,X,Y) - матрично-векторное произведение Y=АX
//автоподбор - матрица ли это из одного массива - DD
//или же это матрица из двух массивов - Aval, AI, AJ

void SpMV(double *Aval, int  *AI, int *AJ, double *X, double *Y, int N){
 
    double sum;
    int i;
    #pragma omp parallel for private(i) reduction(+:sum)
        for (i = 0; i < N; i++) 
        {
            sum = 0.0;
            for (int j = AI[i]; j < AI[i+1]; j++) 
            {
                int m = AJ[j];
                sum += Aval[j] * X[m];
            }
            Y[i] = sum;
        }
    return;
}

void SpMV(double *DD, double *X, double *Y, int N) { 

    int i = 0;

    #pragma omp parallel for private(i) 
        for (i=0; i < N; i++) 
            Y[i] = DD[i] * X[i];
    return;
}

/*----------------------------------------------------------*/
// axpby(PP, RR, betai_1, 1.0);

void axpby(double *X, double *Y, int N, double x1, double x2){
   
    int i;
    #pragma omp parallel for private(i) 
        for(i=0 ; i < N; i++)
            X[i] = x1 * X[i] + x2 * Y[i];
    return;
}

/*----------------------------------------------------------*/
// dot(PP, PP, N);

double dot(double *X, double *Y, int N){

    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
        for(int i=0 ; i < N; i++)
            sum += X[i] * Y[i];
    return sum;    
}

/*----------------------------------------------------------*/
//solver

int solver(int N, double *Aval, int *AI, int *AJ, double *BB, double tol, int maxit){

    double *RR = new double[N];
    repeatmatr(RR, BB, N);

    double *RR2 = new double[N];
    repeatmatr(RR2, BB, N);

    double *PP = new double[N];
    double *PP2 = new double[N];
    double *VV = new double[N];
    double *SS = new double[N];
    double *SS2 = new double[N];
    double *TT = new double[N];
    double *XX = new double[N];
    double *DD = new double[N];

    int i;

    #pragma omp parallel for private(i)
        for(i=0; i<N; i++){
            DD[i] = 0;
            XX[i] = 0;
        }
    makediag(DD, Aval, AI, AJ, N);

    double mineps = 1E-15;
    double initres = sqrt(dot(RR, RR, N));
    double eps = Max(mineps, tol*initres);
    double res = initres;

    double Rhoi_1 = 1.0;
    double alphai = 1.0; 
    double wi = 1.0; 
    double betai_1 = 1.0; 
    double Rhoi_2 = 1.0; 
    double alphai_1 = 1.0;
    double wi_1 = 1.0; 
    double RhoMin = 1E-60;

    int info = 1;

    int I;
    for(I=0; I < maxit; I++){

        if(info) 
            printf("It %d : res = %e tol = %e \n",I, res, res/initres);

        if(res < eps) 
            break;

        if(res > initres / mineps) 
            return -1;

        if(I == 0)
            Rhoi_1 = initres * initres;
        else 
            Rhoi_1 = dot(RR2, RR, N);

        if(fabs(Rhoi_1) < RhoMin) 
            return -1;

        if(I == 0)
            repeatmatr(PP, RR, N);
        else{
            betai_1 = (Rhoi_1 * alphai_1) / (Rhoi_2 * wi_1);
            // p = r + betai_1 * (p - w1 * v)
            axpby(PP, RR, N, betai_1, 1.0); 
            axpby(PP, VV, N, 1.0, -wi_1 * betai_1);
        }

        SpMV(DD, PP, PP2, N);
        SpMV(Aval, AI, AJ, PP2, VV, N);

        alphai = dot(RR2, VV, N);
        if(fabs(alphai) < RhoMin) 
            return -3;
        alphai = Rhoi_1 / alphai;

        // s = r - alphai * v
        repeatmatr(SS, RR, N); 
        axpby(SS, VV, N, 1.0, -alphai);

        SpMV(DD, SS, SS2, N);
        SpMV(Aval, AI, AJ, SS2, TT, N);

        wi = dot(TT, TT, N);
        if(fabs(wi) < RhoMin) 
            return -4;

        wi = dot(TT, SS, N) / wi;
        if(fabs(wi) < RhoMin) 
            return -5;
        // x = x + alphai * p2 + wi * s2
        axpby(XX, PP2, N, 1.0, alphai);
        axpby(XX, SS2, N, 1.0, wi);

        // r = s - wi * t
        repeatmatr(RR, SS, N);
        axpby(RR, TT, N, 1.0, -wi);

        alphai_1 = alphai;
        Rhoi_2 = Rhoi_1;
        wi_1 = wi;

        res = sqrt(dot(RR, RR, N));
    } 

    cout << endl;
    if(info) 
        printf("Solver_BiCGSTAB: outres: %g\n",res);
    
    return I;
} 

/*---------------------------------------------------------------*/
//test_function

void test_dot(int N) {

    double *X = new double[N];
    double *B = new double[N];
    for (int i = 0; i < N; i++) {
        X[i] = sin(i);
        B[i] = sin(i);
    }
    
    int Ntest = 20;
    double tdotseq = 0.0, t, result;
    const double dotflop = Ntest * Ntest * N * 2 * 1E-9;
    
    cout << "(DOT) testing sequential ops:" << endl;
    
    omp_set_num_threads(1);

    for(int i=0; i < Ntest; i++){
        t = omp_get_wtime();
        for(int j=0; j < Ntest; j++) 
            result = dot(X, B, N);
        tdotseq += omp_get_wtime() - t;
    }
    printf("dot time=%6.3fs GFLOPS=%6.2f\n", tdotseq, dotflop/tdotseq);
    
    //parallel mode
    const int NTR = omp_get_num_procs();
    for(int ntr=2; ntr <= NTR; ntr += 2){
        for(int i=0; i < N; i++){ 
            X[i] = 0.0; 
            B[i] = (i * i) % 123; 
        }
        
        cout << "(DOT) testing parallel ops for ntr=" << ntr << ":" << endl;
        
        omp_set_num_threads(ntr);
        double tdotpar = 0.0;
        for(int i=0; i < Ntest; i++){
            t = omp_get_wtime();
            for(int j=0; j < Ntest; j++) 
                result = dot(X, B, N);
            tdotpar += omp_get_wtime() - t;
        }
        printf("dot time=%6.3fs GFLOPS=%6.2f Speedup=%6.2fX \n",
               tdotpar, dotflop/tdotpar, tdotseq/tdotpar);
    }
}

void test_axpby(int N) {
    double *X = new double[N];
    double *B = new double[N];
    double *R = new double[N];

    for (int i = 0; i < N; i++) {
        X[i] = sin(i);
        B[i] = sin(i);
    }
    double alpha = 1.00001, beta = 0.99999;
    
    int Ntest = 20;
    double tdotseq=0.0, t;
    const double dotflop = Ntest * Ntest * N * 3 * 1E-9;
    
    cout << "(AXPBY) testing sequential ops:" << endl;
    
    omp_set_num_threads(1);

    for(int i=0; i < Ntest; i++){
        t = omp_get_wtime();
        for(int j=0; j < Ntest; j++) {
            axpby(X, B, alpha, beta, N);
            repeatmatr(R,X,N);
        }
        tdotseq += omp_get_wtime() - t;
    }
    printf("axpby time=%6.3fs GFLOPS=%6.2f\n", tdotseq, dotflop/tdotseq);
    
    //parallel mode
    const int NTR = omp_get_num_procs();
    for(int ntr=2; ntr <= NTR; ntr+=2){
        for(int i=0; i < N; i++){ 
            X[i] = 0.0; 
            B[i] = (i * i) % 123; 
        }
        cout << "(AXPBY) testing parallel ops for ntr=" << ntr << ":" << endl;
        omp_set_num_threads(ntr);
        double tdotpar = 0.0;
        for(int i=0; i < Ntest; i++){
            t = omp_get_wtime();
            for(int j=0; j < Ntest; j++) {
                axpby(X, B, alpha, beta, N);
                repeatmatr(R,X,N);
            }
            tdotpar += omp_get_wtime() - t;
        }
        printf("axpby time=%6.3fs GFLOPS=%6.2f Speedup=%6.2fX \n",
               tdotpar, dotflop/tdotpar, tdotseq/tdotpar);
    }
}

void test_spmv(int N) {
    double *M = new double[N];
    double *B = new double[N];
    double *R = new double[N];

    for (int i = 0; i < N; i++)
        B[i] = sin(i);
    
    int Ntest = 20;
    double tdotseq = 0.0, t;
    const double dotflop = Ntest * Ntest * N * 2 * 1E-9;
    
    cout << "(SPMV)testing sequential ops: " << endl;
    
    omp_set_num_threads(1);
    for(int i=0; i < Ntest; i++){
        t = omp_get_wtime();
        for(int j=0; j < Ntest; j++) 
            SpMV(R, M, B, N);
        tdotseq += omp_get_wtime() - t;
    }
    printf("spmv time=%6.3fs GFLOPS=%6.2f\n", tdotseq, dotflop/tdotseq);
    
    //parallel mode
    const int NTR = omp_get_num_procs();
    for(int ntr=2; ntr <= NTR; ntr+=2){
        for(int i=0; i < N; i++)
            B[i] = (i * i) % 123;
        cout << " (SPMV) testing parallel ops for ntr=" << ntr << ":" << endl;
        omp_set_num_threads(ntr);
        double tdotpar = 0.0;
        for(int i=0; i < Ntest; i++){
            t = omp_get_wtime();
            for(int j=0; j < Ntest; j++) 
                SpMV(R, M, B, N);
            tdotpar += omp_get_wtime() - t;
        }
        printf("spmv time=%6.3fs GFLOPS=%6.2f Speedup=%6.2fX \n",
               tdotpar, dotflop/tdotpar, tdotseq/tdotpar);
    }
}

void test_function() {
        
    cout << "N: 1000" << endl;
    cout << endl;
    test_dot(10 * 10 * 10);
    cout << endl;
    test_axpby(10 * 10 * 10);
    cout << endl;
    test_spmv(10 * 10 * 10);
    cout << endl; 
    
    cout << "N: 10000" << endl;
    cout << endl;
    test_dot(10 * 100 * 10);
    cout << endl;
    test_axpby(10 * 100 * 10);
    cout << endl;
    test_spmv(10 * 100 * 10);
    cout << endl;

    cout << "N: 100000" << endl;
    cout << endl;   
    test_dot(100 * 10 * 100);
    cout << endl;
    test_axpby(100 * 10 * 100);
    cout << endl;
    test_spmv(100 * 10 * 100);
    cout << endl;

    cout << "N: 1000000" << endl;
    cout << endl;   
    test_dot(100 * 100 * 100);
    cout << endl;
    test_axpby(100 * 100 * 100);
    cout << endl;
    test_spmv(100 * 100 * 100);
    cout << endl;
}

/*-----------------------------------------------------------------------------*/

int main(int argc, char *argv[]){

    int Nx = atoi(argv[1]);
    int Ny = atoi(argv[2]);
    int Nz = atoi(argv[3]);
   
    unsigned int N = Nx * Ny * Nz;

    double eps   = atof(argv[4]);
    int    maxit = atoi(argv[5]);
    int    qa    = atoi(argv[6]);

    int i=0, j=0;

    double *Aval = new double[7*N];
    int *AJ = new int[7*N];
    int *AI = new int[N+1];

    AI[0] = 0;
    int Aval_counter = 0;
    int AI_counter = 0;
    int AJ_counter = 0;

    for(int K = 0; K < Nz; K++){
        for(int J = 0; J < Ny; J++){
            for(int I = 0; I < Nx; I++){

                i = K * (Nx * Ny)+ J * Nx + I;
            
                double sum = 0;
                int counter = 0;

                if(i < N){

                    if (K > 0){
                        j = i - Nx * Ny;
                        if(j < N){
                            Aval[Aval_counter++] = sin(i + j + 1);
                            AJ[AJ_counter++] = j;
                            sum = sum + fabs(sin(i + j + 1));
                            counter++;
                        }
                    }
                    
                    if (J > 0){
                        j = i - Nx;
                        if(j < N){
                            Aval[Aval_counter++] = sin(i + j + 1);
                            AJ[AJ_counter++] = j;
                            sum = sum + fabs(sin(i + j + 1));
                            counter++;
                        }
                    }

                    if (I > 0){
                        j = i - 1;
                        if(j < N){
                            Aval[Aval_counter++] = sin(i + j + 1);
                            AJ[AJ_counter++] = j;
                            sum = sum + fabs(sin(i + j + 1));
                            counter++;
                        }
                    }

                    int num = Aval_counter;
                    Aval[Aval_counter++] = 0;
                    AJ[AJ_counter++] = i;

                    if (I < Nx - 1){
                        j = i + 1;
                        if(j < N){
                            Aval[Aval_counter++] = sin(i + j + 1);
                            AJ[AJ_counter++] = j;
                            sum = sum + fabs(sin(i + j + 1));
                            counter++;
                        }
                    }

                    if (J < Ny - 1){
                        j = i + Nx;
                        if(j < N){
                            Aval[Aval_counter++] = sin(i + j + 1);
                            AJ[AJ_counter++] = j;
                            sum = sum + fabs(sin(i + j + 1));
                            counter++;
                        }
                    }

                    if (K < Nz - 1){
                        j = i + Nx * Ny;
                        if(j < N){
                            Aval[Aval_counter++] = sin(i + j + 1);
                            AJ[AJ_counter++] = j;
                            sum = sum + fabs(sin(i + j + 1));
                            counter++;
                        } 
                    }

                    Aval[num] = 1.1 * sum;
                    counter++;
                    AI[AI_counter+1] = AI[AI_counter] + counter;
                    AI_counter++;
                }
           }
       }
    }

    AI_counter++;
    AI[AI_counter] = AJ_counter;

    double *BB = new double[N];
    for(i = 0; i < N; i++)
        BB[i] = sin(i);
    double initres = sqrt(dot(BB, BB, N));

    double tol = eps / initres;

    cout << endl;
    cout << "Testing BiCGSTAB solver for a 3D grid domain : " << endl;
    cout << "N = "<<N<<"(Nx="<<Nx<<", Ny="<<Ny<<", Nz="<<Nz<<")"<<endl;
    cout << "Aij = sin((double)i+j+1), i!=j" << endl;
    cout << "Aii = 1.1*sum(fabs(Aij))" << endl;
    cout << "Bi  = sin((double)i)" << endl;
    cout << "tol = " << tol << endl;
    cout << endl;

    int nit = solver(N, Aval, AI, AJ, BB, tol, maxit);
    printf("Solver finished in %d iterations, tol = %e", nit, tol);
    cout << "\n" << endl;
    cout << "------------------------------------------------------ " << endl;
    cout << endl;
    cout << "Testing functions : " << endl;
    cout << endl;

    if (qa) {
        test_function();  
    }

    return 0;
        
}
    