#include <iostream>
#include <filesystem>
#include <unordered_map>
#include <cmath>
#include "ModelR1.h"

float alpha_val = 0.0035f;  // α = 0.0035
float beta_val  = 0.00001f;     // β
float delta_val = 0.000065f;     // δ

double cost_function(const Eigen::VectorXf& x) {
    return x(0)*x(0)
            + alpha_val* x(1)*x(1)
            + beta_val * x(2)*x(2)
            + delta_val* x(3)*x(3);
}

static std::unordered_map<int, Eigen::VectorXf> Goal;

Eigen::VectorXf goal(double current_time) {
    return Goal[std::round(current_time/0.02)];
}

bool stopping_criterion(const Eigen::VectorXf& x) {
    return false;
}

int main() {
    // Monta a trajetória desejada em Goal
    float f = 1.0f;
    for (float time = 0.0f; time <= 40.0f; time += 0.02f) {
        Eigen::VectorXf axx(4);
        axx << std::cos(time), -f*std::sin(time), std::sin(time), f*std::cos(time);
        Goal[std::round(time/0.02)] = axx;
    }

    // Parâmetros do sistema
    const int n = 4, m = 1, p = 4;
    Eigen::MatrixXf A(n,n), B(n,m), C(p,n), D(p,m), discretization_actions(m,3);
    Eigen::VectorXf noise(m);

    A <<  1.0f, 0.02f,     -3.27345e-06f, 5.00671e-05f,
          0.0f, 1.0f,     -0.000491158f, 0.00500924f,
          0.0f, 0.0f,      0.998035f,     0.020037f,
          0.0f, 0.0f,     -0.196563f,     1.00304f;
    B <<  0.00196527f,
          0.196691f,
          0.00196463f,
          0.196563f;
    C.setIdentity();
    D.setZero();
    noise << 0.03f;
    int actions_state_possible = 4;
    float discrete_time_step = 0.02f;
    discretization_actions << -0.2f, 0.2f, 0.04f;

    Eigen::VectorXf x_0(n);
    x_0 << 0, 0, 0, 0;

    // Cria o modelo
    Model model1(
        actions_state_possible,
        A, B, C, D,
        discretization_actions,
        noise,
        cost_function,
        goal,
        discrete_time_step
    );

    int horizon = 5;

    // ** NOVO: configura pasta de saída e nome de arquivo começando com "Elliptic2" **
    namespace fs = std::filesystem;
    fs::path outputDir = R"(C:\Supaero\Stage\Codigo3\Analysis\Data2\Elliptic)";
    if (!fs::exists(outputDir)) {
        fs::create_directories(outputDir);
        std::cout << "Criada pasta: " << outputDir << std::endl;
    }

    // Define o prefixo "Elliptic2" + sufixo de sua escolha
    std::string simBaseName = "Elliptic2DD";
    std::string simFullPath = (outputDir / simBaseName).string();

    // Inicia a simulação gravando CSVs em:
    // C:\Supaero\Stage\Codigo3\Analysis\Data2\Elliptic\Elliptic2..*.csv
    model1.startSimulation(
        simFullPath,       // base do arquivo de saída
        20,                // número de iterações
        x_0,               // estado inicial
        5,                 // parâmetro adicional
        horizon,           // horizonte de planejamento
        stopping_criterion // critério de parada
    );

    return 0;
}
