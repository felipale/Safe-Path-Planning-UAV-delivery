// -----------------------------------------------------------------------------
// Este arquivo define a classe `c2d`, que transforma um sistema contínuo linear
// com matrizes (A, B, C, D) em sua forma discretizada (A_d, B_d, C_d, D_d)
// usando aproximações por séries de Taylor.
// 
// A discretização é feita para permitir o uso de técnicas de planejamento
// e controle em tempo discreto, como MPC e métodos baseados em MDP.
// -----------------------------------------------------------------------------

#include <iostream>
#include <cmath>
#include <list>
#include <algorithm>
#include <random>
#include <chrono>
#include <vector>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <functional>
#include "Myhash.h"


// Usamos o namespace padrão para evitar std::
using namespace std;

// Classe para discretização de um sistema linear contínuo
class c2d{
    public:
        
        Eigen::MatrixXf A_dis, B_dis, C_dis, D_dis;                                                 // Matrizes do sistema discretizado
        double t_dis;                                                                               // Período de amostragem

        // Construtor que recebe o sistema contínuo A,B,C,D e o tempo de amostragem t
        c2d(Eigen::MatrixXf A, Eigen::MatrixXf B, Eigen::MatrixXf C, Eigen::MatrixXf D, double t){
            this->t_dis = t;            // Armazena o tempo de discretização
            this->C_dis = C;            // C e D são diretamente copiadas
            this->D_dis = D;
            calculateA_dis(A);          // Discretiza A
            calculateB_dis(A,B);        // Discretiza B

        }

        // Discretização da matriz A por aproximação de série de Taylor (exp(A*t))
        void calculateA_dis(Eigen::MatrixXf A){
            Eigen::MatrixXf Ax(A.rows(), A.rows()), Axx(A.rows(), A.rows());
            Ax.setIdentity(A.rows(), A.rows());                                 // Inicia Ax como identidade
            int n = 1;
            Axx = t_dis* A;                                                     // Primeiro termo da série

            // Expansão da exponencial de matriz: e^(At) ≈ I + At + (At)^2/2! + (At)^3/3! + ...
            for(n = 2; n <20; n++){
                Ax += Axx; 
                Axx *= (t_dis/n)* A;                                            // Próximo termo: (At)^n / n!                                    
            }
        this->A_dis = Ax;                                                       // A discretizado
        }

        // Discretização da matriz B
        // Em teoria, B_dis = ∫₀^T e^{Aτ} dτ · B ≈ (série de Taylor com correções)
        void calculateB_dis(Eigen::MatrixXf A, Eigen::MatrixXf B){
            Eigen::MatrixXf Ax(A_dis.rows(), A_dis.rows()), Axx(A_dis.rows(), A_dis.rows());
            
            // Inicia Ax como identidade vezes o passo de tempo (termo linear do integrando)
            Ax.setIdentity(A_dis.rows(), A_dis.rows());
            Ax *= t_dis;
            
            
            int n = 1;

            // Segundo termo da série para a integral: (A·t)^2 / 2!
            Axx = (t_dis* t_dis)*A*1/2;
            
            // Aproxima a integral de e^{At} com uma série de termos negativos
            for(n = 2; n <20; n++){
                Ax -= Axx; 
                Axx *= (-t_dis/(n+1))* A;
            }
        // Multiplica pela matriz B e pela aproximação da integral de e^{At}
        this->B_dis = A_dis*Ax*B;
        }
        // ⚠️ Teoricamente, o correto seria: B_dis = ∫₀^T e^{Aτ} dτ · B
        // Aqui foi feita uma aproximação via série que pode não ser exatamente isso,
        // então este ponto pode ser revisado numericamente com expm(A·t).
        
        
        // Função para exibir as matrizes discretizadas
        void display(){
            cout << "A_dis:"<< A_dis<< endl;
            cout << "B_dis:"<< B_dis<< endl;
            cout << "C_dis:"<< C_dis<< endl;
            cout << "D_dis:"<< D_dis<< endl;

        }

};



int main(){
    const int n =2;         // Número de estados
    const int m = 1;        // Número de entradas (1 atuador)
    const int p = 2;
    Eigen::MatrixXf A(n,n);
    Eigen::MatrixXf B(n,m);
    Eigen::MatrixXf C(p,n);
    Eigen::MatrixXf D(p,m);
   
    
    
    A << 0, 50,
         0, 0;
    

    B << 0,
         5638;


    C << 1, 0,
         0, 1; 
    
    D << 0,
         0;

    c2d discrete(A, B, C, D, 0.02);
    discrete.display();
    return 0;
}