#include "ModelR2.h"
#include <filesystem>
#include <iostream>
#include <unordered_map>
#include <cmath>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// função de custo
double cost_function(const VectorXf& x) {
    double a = x(0)*x(0)
             + 0.0035 * x(1)*x(1)
             +        x(4)*x(4)
             + 0.0035 * x(5)*x(5);
    if (fabs(x(1)*0.0001f) > 0.5f) {
        return a * 30.0;
    } else {
        return a;
    }
}

// tabela de referência
static unordered_map<int, VectorXf> Goal;

VectorXf goalx(double current_time) {
    int idx = int(round(current_time / 0.02));
    return Goal.at(idx).segment(0,2);
}

VectorXf goaly(double current_time) {
    int idx = int(round(current_time / 0.02));
    return Goal.at(idx).tail(2);
}

bool stopping_criterion(const VectorXf& /*x*/) {
    return false;
}

int main() {
    // força o diretório de saída
    filesystem::current_path("C:\\Supaero\\Stage\\Codigo3\\Analysis\\DataFinal\\ElipticUnCoupled");
    cout << "Salvando resultados em: " << filesystem::current_path() << endl;

    // constrói a tabela de referência (8-dim vetor)
    const float f = 1.0f;
    VectorXf axx(8);
    for (float t = 0.0f; t <= 120.0f; t += 0.02f) {
        axx(0) = cosf(t);
        axx(1) = -f * sinf(t);
        axx(2) = 0.0f;
        axx(3) = 0.0f;
        axx(4) = sinf(t);
        axx(5) =  f * cosf(t);
        axx(6) = 0.0f;
        axx(7) = 0.0f;
        Goal[int(round(t / 0.02f))] = axx;
    }

    // dimensões
    const int n = 4, m = 1, p = 4;

    // declarações
    MatrixXf A(n,n), B(n,m), C(p,n), D(p,m), discretization_actions(m,3);
    VectorXf noise(m), x_0(2*n);

    // inicializa A, B, C, D
    A <<  1.0f,   0.02f,      -3.27345e-06f, 5.00671e-05f,
          0.0f,   1.0f,       -0.000491158f, 0.00500924f,
          0.0f,   0.0f,        0.998035f,      0.020037f,
          0.0f,   0.0f,       -0.196563f,      1.00304f;

    B.setZero();
    B(0,0) = 0.00196527f;
    B(1,0) = 0.196691f;
    B(2,0) = 0.00196463f;
    B(3,0) = 0.196563f;

    C << 1,0,0,0,
         0,1,0,0,
         0,0,1,0,
         0,0,0,1;

    D.setZero();

    // discretização de ações [min, max, passo]
    discretization_actions.setZero();
    discretization_actions(0,0) = -0.2f;
    discretization_actions(0,1) =  0.2f;
    discretization_actions(0,2) =  0.04f;

    // estado inicial e ruído
    x_0.setZero();          // vetor 8×1
    noise.setZero(); noise(0) = 0.03f;

    // parâmetros do MPC
    const int    actions_state_possible = 4;
    const float  discrete_time_step      = 0.02f;

    // cria e executa a simulação
    Model model1(
        actions_state_possible,
        A, B, C, D,
        discretization_actions,
        noise,
        cost_function,
        goalx,
        goaly,
        discrete_time_step
    );

    model1.startSimulation(
        "2DelpticUncoupled",  // nome
        60,                   // tempo total [s]
        x_0,                  // estado inicial
        5,                    // horizonte de controle
        5,                    // horizonte de predição
        stopping_criterion
    );

    return 0;
}
