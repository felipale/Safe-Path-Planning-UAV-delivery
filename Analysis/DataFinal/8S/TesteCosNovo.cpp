#include <Eigen/Dense>
#include "ModelR1.h"
#include <filesystem>
#include <cmath>
#include <unordered_map>
#include <iostream>

double cost_function(const Eigen::VectorXf& x) {
    return x(0)*x(0) + 0.0035f*x(1)*x(1) + 0.0001f*x(2)*x(2) + 0.000065f*x(3)*x(3);
}

static std::unordered_map<int, Eigen::VectorXf> Goal;
static int last_index = 0;  // índice máximo populado em Goal

Eigen::VectorXf goal(double current_time) {
    int idx = std::lround(current_time / 0.02);
    if (idx < 0) idx = 0;
    if (idx > last_index) idx = last_index;
    return Goal[idx];
}

bool stopping_criterion(const Eigen::VectorXf& x) {
    return false;  // sem parada antecipada
}

int main() {

    // força o diretório de saída para C:\Supaero\Stage\...
    std::filesystem::current_path("C:\\Supaero\\Stage\\Codigo3\\Analysis\\DataFinal\\8S");
    std::cout << "Salvando resultados em: " 
            << std::filesystem::current_path() 
            << std::endl;
    // Parâmetros de tempo
    const float dt      = 0.02f;
    const float t_final = 10.0f;   // simulação até 14 segundos

    // Definição de PI
    constexpr double PI = 3.14159265358979323846;

    // Preenche Goal com x₀(t) = cos(2π·0.1·t), outras componentes zero
    Eigen::VectorXf axx(4);
    for (float t = 0.0f; t <= t_final; t += dt) {
        double y = std::cos(2.0 * PI * 0.1 * t);  // cosseno 0.1 Hz
        axx << static_cast<float>(y), 0.0f, 0.0f, 0.0f;
        int idx = std::lround(t / dt);
        Goal[idx] = axx;
        last_index = idx;
    }

    // Definição do sistema
    const int n = 4, m = 1, p = 4;
    Eigen::MatrixXf A(n,n), B(n,m), C(p,n), D(p,m), discretization_actions(m,3);
    Eigen::VectorXf noise(m);

    A <<  1,         0.02f,       -3.27345e-06f,  5.00671e-05f,
          0,            1.0f,      -0.000491158f,   0.00500924f,
          0,            0,         0.998035f,       0.020037f,
          0,            0,        -0.196563f,       1.00304f;

    B << 0.00196527f,
         0.196691f,
         0.00196463f,
         0.196563f;

    C << 1,0,0,0,
         0,1,0,0,
         0,0,1,0,
         0,0,0,1;

    D << 0,0,0,0;

    // Ações discretas em [–0.2, 0.2] com passo 0.05
    discretization_actions << -0.2f, 0.2f, 0.05f;

    noise << 0.03f;

    // Estado inicial
    Eigen::VectorXf x0(n);
    x0 << 0,0,0,0;

    // Cria modelo
    int actions_state_possible = 4;
    Model model1(
        actions_state_possible,
        A, B, C, D,
        discretization_actions,
        noise,
        cost_function,
        goal,
        dt
    );

    // Executa simulação
    int horizon = 7;
    model1.startSimulation("CosineTrackingABD", 10, x0,4, horizon, stopping_criterion);

    return 0;
}