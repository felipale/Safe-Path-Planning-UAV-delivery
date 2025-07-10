#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

// Definição da classe para o modelo do drone com carga
class DronePayloadModel {
public:
    // Estados do sistema
    Vector4f state;  // [posX, velX, posY, velY, angulo_cargaX, velocidade_angularX, angulo_cargaY, velocidade_angularY]

    // Parâmetros físicos do drone e carga
    float m_drone;  // massa do drone
    float m_payload;  // massa da carga
    float l_cable;  // comprimento do cabo
    float g;  // aceleração da gravidade

    // Matrizes de estado (para um modelo linearizado simplificado)
    Matrix4f A;
    Vector4f B;

    // Construtor
    DronePayloadModel(float md, float mp, float l, float gravity) {
        m_drone = md;
        m_payload = mp;
        l_cable = l;
        g = gravity;

        // Inicialização do vetor de estados
        state = Vector4f::Zero();

        // Inicialização das matrizes (exemplo simplificado)
        A = Matrix4f::Zero();
        B = Vector4f::Zero();

        setDynamics();
    }

    // Configuração das matrizes A e B para o modelo linearizado simplificado
    void setDynamics() {
        // 1. Relações cinemáticas básicas entre posições e velocidades
        // Zerar as matrizes
        A.setZero();
        B.setZero();

        // Constantes intermediárias
        float M = m_drone + m_payload;
        float k = (m_payload * l_cable) / M;
        float gamma = m_payload / M;
        float a = g; // Assumindo hover (T ≈ Mg) -- Formula completa: T/M = M*g/M
        float b = 1 / l_cable;

        // Matriz A - matriz dinâmica do sistema
        A(0, 1) = 1.0;
        A(1, 3) = k;
        A(2, 3) = 1.0;
        A(3, 2) = -g / l_cable;
        A(3, 3) = -gamma;

        // Matriz B - influência direta do comando de pitch (theta)
        B(1) = a;
        B(3) = a * b;
    }

    // Método para atualização do estado com base no modelo
    void updateState(float tetha, float dt) {
        // Equação do modelo discretizado
        state += dt * (A * state + B * tetha);
    }

    // Método para imprimir o estado atual do sistema
    void printState() const {
        cout << "Estado atual do drone e carga:\n" << state << endl;
    }
};


int main()
{
    DronePayloadModel model(1.5f, 0.5f, 1.0f, 9.81f);
    
    float thetaCmd = 0.05f; // rad
    float dt = 0.01f;

    for (int k=0; k<1000; ++k)
        model.updateState(thetaCmd, dt);

    model.printState();
    return 0;
}
