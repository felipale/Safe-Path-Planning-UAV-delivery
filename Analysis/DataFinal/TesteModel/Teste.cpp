#include <iostream>
#include <filesystem>
#include <unordered_map>
#include <cmath>
#include "ModelR1.h"
#include <Eigen/Dense>

// Pesos de custo: penalizam velocidades e ângulos em X e Y
float alpha_val = 0.0035f;   // peso para \dot x, \dot y (mesmo para ambos)
float beta_val  = 0.0001f;    // peso para \alpha, \beta (≥1.7e-3 conforme relatório)
float delta_val = 0.000065f;    // peso para \dot\alpha, \dot\beta (≥2.5e-3 conforme relatório)

// Mapa de metas (8 estados): [x, ẋ, α, α̇,  y, ẏ, β, β̇]
static std::unordered_map<int, Eigen::VectorXf> Goal;

Eigen::VectorXf goal(double current_time) {
    int k = int(std::round(current_time / 0.02));
    return Goal[k];
}

// Função de custo: não penaliza posição, apenas velocidades e ângulos dos dois eixos
double cost_function(const Eigen::VectorXf& x) {
    return
        alpha_val * x(1)*x(1)
      + beta_val  * x(2)*x(2)
      + delta_val * x(3)*x(3)
      + alpha_val * x(5)*x(5)
      + beta_val  * x(6)*x(6)
      + delta_val * x(7)*x(7);
}

bool stopping_criterion(const Eigen::VectorXf& /*x*/) {
    return false;
}

int main() {
    // 1) Monta a trajetória desejada em Goal (8 estados)
    float f = 1.0f;
    for (int k = 0; k <= int(40.0f/0.02f); ++k) {
        double t = k * 0.02;
        Eigen::VectorXf g(8);
        // posição e velocidade linear desejadas
        g(0) =  std::cos(t);
        g(1) = -f * std::sin(t);
        g(4) =  std::sin(t);
        g(5) =  f * std::cos(t);
        // referências de ângulo e velocidade angular
        // TODO: ajustar se diferente da linear
        g(2) =  std::sin(t);
        g(3) =  f * std::cos(t);
        g(6) =  std::sin(t);
        g(7) =  f * std::cos(t);    // β̇_ref(t)
        Goal[k] = g;
    }

    // 2) Parâmetros do sistema discreto
    const int n = 8, m = 2, p = 8;
    Eigen::MatrixXf A(n,n);                // 8×8
    Eigen::Matrix<float,8,2> B;            // 8×2
    Eigen::MatrixXf C(p,n), D(p,m);
    Eigen::Matrix<float,2,3> discretization_actions;
    Eigen::VectorXf noise(m);

    // 2.1) Bloco diagonal com A_4×4 já discretizado (substitua pelos seus valores)
    Eigen::Matrix4f A4;
    A4 <<  1.0f, 0.02f,     -3.27345e-06f, 5.00671e-05f,
           0.0f, 1.0f,     -0.000491158f, 0.00500924f,
           0.0f, 0.0f,      0.998035f,     0.020037f,
           0.0f, 0.0f,     -0.196563f,     1.00304f;
    A.setZero();
    A.block<4,4>(0,0) = A4;  // X
    A.block<4,4>(4,4) = A4;  // Y

    // 2.2) B discretizado para θ (X) e φ (Y)
    Eigen::Vector4f B4_x;
    B4_x << 0.00196527f,
            0.196691f,
            0.00196463f,
            0.196563f;
    Eigen::Vector4f B4_y;
    // TODO: preencha B4_y com valores discretizados para φ (eixo Y)
    B4_y = -B4_x;

    B.setZero();
    B.block<4,1>(0,0) = B4_x;    // entrada θ afeta sub-sistema X
    B.block<4,1>(4,1) = B4_y;    // entrada φ afeta sub-sistema Y

    C.setIdentity();
    D.setZero();

    noise << 0.03f, 0.03f;        // ruído em cada entrada

    // 2.3) discretização das ações: [min, max, passo] para θ e φ
    discretization_actions <<
        -0.2f, 0.2f, 0.04f,   // θ
        -0.2f, 0.2f, 0.04f;   // φ

    // Estado inicial (8 estados)
    Eigen::VectorXf x_0(n);
    x_0.setZero();

    // 3) Cria o modelo com m=2 entradas
    Model model(
        /*actions_state_possible=*/11,  // ajuste conforme discretization_actions
        A, B, C, D,
        discretization_actions,
        noise,
        cost_function,
        goal,
        /*dt=*/0.02f
    );


    // 4) Simulação
    int horizon = 10;  // horizonte maior para capturar divergências

    namespace fs = std::filesystem;
    fs::path outputDir = R"(C:\Supaero\Stage\Codigo3\Analysis\DataFinal\TesteModel)";
    if (!fs::exists(outputDir)) {
        fs::create_directories(outputDir);
        std::cout << "Criada pasta: " << outputDir << std::endl;
    }

    std::string simBaseName = "Elliptic2DD";
    std::string simFullPath = (outputDir / simBaseName).string();

    model.startSimulation(
        simFullPath,
        /*Tempo =*/20,
        x_0,
        /*Trials=*/5,
        horizon,
        stopping_criterion
    );

    return 0;
}
