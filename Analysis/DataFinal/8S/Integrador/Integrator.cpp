#include <Eigen/Dense>
#include "ModelR1.h"
#include <filesystem>
#include <cmath>
#include <unordered_map>
#include <iostream>

// ------- Parâmetros de peso -------
static const float alfa1 = 1.0f;      // posição
static const float alfa2 = 0.0035f;   // velocidade
static const float alfa3 = 0.000f;   // ângulo da carga
static const float alfa4 = 0.000f; // velocidade angular da carga
static const float alfaI = 0.001f;      // peso do integrador

// Função de custo que agora penaliza o estado integrador x(4)
double cost_function(const Eigen::VectorXf& x) {
    // x.size() == 5
    double J  = alfa1 * x(0)*x(0)
               + alfa2 * x(1)*x(1)
               + alfa3 * x(2)*x(2)
               + alfa4 * x(3)*x(3)
               + alfaI * x(4)*x(4);  // termo integrador
    return J;
}

// Goal: mapeia índice de passo a vetor de estados referencial (5 estados)
static std::unordered_map<int, Eigen::VectorXf> Goal;
static int last_index = 0;

Eigen::VectorXf goal(double current_time) {
    int idx = std::lround(current_time / 0.02);
    if (idx < 0) idx = 0;
    if (idx > last_index) idx = last_index;
    return Goal[idx];
}

bool stopping_criterion(const Eigen::VectorXf& x) {
    return false;  // sem critério de parada antecipada
}

int main() {
    // Diretório de saída
    std::filesystem::current_path("C:\\Supaero\\Stage\\Codigo3\\Analysis\\DataFinal\\8S\\Integrador");
    std::cout << "Salvando resultados em: " << std::filesystem::current_path() << std::endl;

    // Parâmetros de tempo
    const float dt      = 0.02f;
    const float t_final = 10.0f;
    constexpr double PI = 3.14159265358979323846;

    // Preenche Goal com [cos(2π·0.1·t), 0,0,0, 0]
    Eigen::VectorXf axx(5);
    for (float t = 0.0f; t <= t_final; t += dt) {
        float y = std::cos(2.0f * PI * 0.1f * t);
        axx << y, 0.0f, 0.0f, 0.0f, 0.0f;
        int idx = std::lround(t / dt);
        Goal[idx] = axx;
        last_index = idx;
    }

    // --- Definição do sistema augmentado (5 estados) ---
    const int n = 5, m = 1, p = 4;
    Eigen::Matrix<float,5,5> A;
    Eigen::Matrix<float,5,1> B;
    Eigen::Matrix<float,4,5> C;
    Eigen::Matrix<float,4,1> D;
    Eigen::Matrix<float,1,3> discretization_actions;
    Eigen::VectorXf noise(m);

    // Matrizes originais (4×4 A_orig, 4×1 B_orig)
    Eigen::Matrix<float,4,4> A_orig;
    Eigen::Matrix<float,4,1> B_orig;
    A_orig <<  1,         dt,       -3.27345e-06f,  5.00671e-05f,
               0,         1.0f,      -0.000491158f,   0.00500924f,
               0,         0,         0.998035f,       0.020037f,
               0,         0,        -0.196563f,       1.00304f;
    B_orig << 0.00196527f,
              0.196691f,
              0.00196463f,
              0.196563f;

    // Monta A_aug
    A.setZero();
    A.block<4,4>(0,0) = A_orig;
    A(4,0) = dt;      // integrado: ξₖ₊₁ = ξₖ + dt*(x₀ₖ − x₀_refₖ)
    A(4,4) = 1.0f;

    // Monta B_aug
    B.setZero();
    B.block<4,1>(0,0) = B_orig;
    // B(4,0) já é zero

    // Saída C (apenas as 4 primeiras componentes)
    C.setZero();
    C.block<4,4>(0,0) = Eigen::Matrix<float,4,4>::Identity();

    // D = zeros
    D.setZero();

    // Ações discretas e ruído
    discretization_actions << -0.2f, 0.2f, 0.05f;
    noise << 0.03f;

    // Estado inicial de 5
    Eigen::VectorXf x0(n);
    x0 << 0, 0, 0, 0,  // originais
          0;           // integrador inicia em zero

    // Cria modelo com o novo número de estados
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

    // Executa simulação: horizonte de 7 passos
    int horizon = 7;
    model1.startSimulation("CosineInteg3", 10, x0, 4, horizon, stopping_criterion);

    return 0;
}
