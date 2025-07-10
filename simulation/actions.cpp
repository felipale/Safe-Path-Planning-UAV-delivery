// -----------------------------------------------------------------------------
// Este arquivo define funções auxiliares usadas no planejamento de trajetória:
// 
// - `cost_function`: avalia o custo de um estado (usado para decisões ótimas).
// - `goal`: retorna a posição objetivo desejada para um dado instante de tempo.
// - `stopping_criterion`: define uma condição para parar o algoritmo (não usada).
// 
// Além disso, o `main()` preenche o mapa `Goal` com os objetivos de trajetória,
// utilizado durante a simulação para definir sub-metas ao longo do tempo.
// -----------------------------------------------------------------------------

#include "ModelR1.h"

double cost_function(const Eigen::VectorXf& x){
    
    Eigen::VectorXf relative;
    // Função de custo quadrático: penaliza erro de posição (x(0)) e velocidade (x(1))
    // Peso 0.0035 corresponde à influência do termo da velocidade no custo total
    return x(0)*x(0) + 0.0035*x(1)*x(1);
}



unordered_map<int, Eigen::VectorXf> Goal;

Eigen::VectorXf goal(double current_time){
     // Retorna a meta (posição desejada) correspondente ao tempo atual discretizado
     // return Goal[round(Ccurrent_time*100)];
    return Goal[round(current_time/0.02)];
}

bool stopping_criterion(const Eigen::VectorXf& x){
// Critério de parada desativado — pode ser usado para detectar convergência
//    Eigen::VectorXf e;
//    e = goal(current_time)  - x;
   return false;//e.maxCoeff() < 0.005;
}

int main(){

    float time;
     Eigen::VectorXf axx(2);
     // Define objetivo 0 até 1 segundo
     for(time = 0; time<=1; time+=0.02){
          // cout << time << endl;
          axx<< 0,0;                         // Posição 0, velocidade 0
          Goal[round(time/0.02)] = axx;
     }

     // Define objetivo 1 a partir de 1s até 60s
     for(time = 1+0.02 ; time<=60; time+=0.02){
          // cout << time << endl;
          axx << 1,0;                        // Posição 1, velocidade 0
          Goal[round(time/0.02)] = axx;
     }

    const int n =2;
    const int m = 1;
    const int p = 2;
    Eigen::MatrixXf A(n,n);
    Eigen::MatrixXf B(n,m);
    Eigen::MatrixXf C(p,n);
    Eigen::MatrixXf D(p,m);
    Eigen::MatrixXf discretization_actions(m,3); //cols (Min, max, level of discretization)
    Eigen::VectorXf noise(m);
    
    
    A << 1, 0.02,
         0, 1;
    

    B << 0.001962,
         0.1962;


    C << 1, 0,
         0, 1; 
    
    D << 0,
         0;

    

    noise<< 0.03;
    int actions_state_possible = 10;
    float discrete_time_step = 0.02;
    discretization_actions << -0.2, 0.2, 0.04;
    Eigen::VectorXf x_0(n);
    x_0 << 0,0;
    
    
    
    int horizon =5;
   
   

    string name = "1DTeste24";
    stringstream ss;
    int pp = 0;

    for (pp = 2; pp< 12; pp++ ){
        cout << pp << endl;
        ss.str("");
	    ss.clear();
        ss  <<"actionsNoise4"<<pp;
        name = ss.str();
        Model model1(pp,A, B, C, D, discretization_actions, noise, cost_function, goal, 0.02);
        model1.startSimulation(name, 5, x_0, 1, horizon, stopping_criterion);
      
    }

    return 0;
}