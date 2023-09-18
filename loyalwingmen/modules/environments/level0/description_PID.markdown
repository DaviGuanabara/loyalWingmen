# Documentação: Controlador PID

## Introdução

O controlador PID (Proporcional, Integral e Derivativo) é um dos controladores de feedback mais utilizados na indústria e na pesquisa devido à sua eficácia e simplicidade. Ele pode ser usado em uma ampla variedade de sistemas de controle para alcançar a performance desejada.

## Design do Controlador

### 1. **Escolha dos Parâmetros**

A classe `PID` foi projetada para ser flexível e facilmente configurável:

- **kp, ki, kd**: São os ganhos Proporcional, Integral e Derivativo, respectivamente. Estes parâmetros determinam a resposta do controlador. A sintonia correta destes parâmetros é crucial para a performance do controlador.
- **set_point**: É o valor desejado pelo usuário.
- **output_limits**: Estes são os limites mínimos e máximos para a saída do controlador, prevenindo saturação.
- **anti_windup**: Opção para habilitar a prevenção de windup no termo integral.
- **integral_limits**: Limites para o termo integral, uma estratégia adicional de anti-windup.
- **deadband**: Região ao redor do set_point onde erros são considerados zero. Útil para sistemas com ruído ou pequenas flutuações.

### 2. **Malha Fechada e Feedback**

O controlador PID opera em malha fechada, o que significa que ele toma decisões com base no erro entre o valor desejado (`set_point`) e o valor atual do sistema (`current_value`). O erro é então usado para calcular a saída do controlador usando os termos P, I e D.

### 3. **Conceitos de Design Empregados**

- **Abstração**: A classe PID encapsula todos os detalhes internos, expondo apenas métodos e propriedades relevantes ao usuário.
- **Flexibilidade**: Os métodos como `update_gains` e `update_set_point` permitem que o usuário altere o comportamento do controlador em tempo real.
- **Extensibilidade**: Usando métodos como `from_config`, a classe pode ser facilmente estendida ou integrada a sistemas mais complexos.
- **Encapsulamento**: Detalhes de implementação, como cálculo de derivada e limitação de saída, são mantidos privados.

## Exemplo de Uso

```python
# Criar uma instância do controlador PID
controller = PID(kp=1.0, ki=0.5, kd=0.01, set_point=100)

# Atualizar o valor desejado
controller.update_set_point(150)

# Em cada loop de controle:
current_value = read_system_value() # Função fictícia para ler o valor atual do sistema
output = controller.compute(current_value, dt=0.01)
apply_output_to_system(output) # Função fictícia para aplicar a saída do controlador ao sistema
```

## Esboço das Funções

- **`compute`**: Calcula a saída do controlador baseada no valor atual e na diferença de tempo desde a última chamada.
- **`is_active`**: Retorna o estado atual (ativo/inativo) do controlador.
- **`update_set_point`**: Atualiza o valor desejado (`set_point`) do controlador.
- **`from_config`**: Método de classe para criar uma instância do controlador a partir de um dicionário de configuração.
- **`toggle`**: Alterna o estado ativo/inativo do controlador.
- **`reset`**: Reinicia os estados internos do controlador.
- **`update_gains`**: Atualiza os ganhos Proporcional, Integral e Derivativo do controlador.

Esperamos que esta documentação forneça uma compreensão clara do design e da operação do controlador PID. A sintonia correta e a compreensão dos conceitos subjacentes são cruciais para a operação eficaz deste controlador em qualquer sistema.

# PIDAutoTuner: Sintonizador Automático de PID

O `PIDAutoTuner` é uma classe destinada a fornecer uma sintonização automática para controladores PID. Ele é baseado no método de Ziegler-Nichols para sistemas de primeira ordem sem tempo morto.

## Teoria

O método de Ziegler-Nichols baseia-se na resposta ao degrau do sistema. A partir dessa resposta, determina-se:

- **K**: A inclinação da tangente no início da curva de resposta ao degrau.
- **T**: O tempo que a resposta leva para atingir 63,2% do seu valor final após a aplicação do degrau.

Os ganhos do PID são então calculados como:

1. \( K_p = 0.6 \times \frac{T}{K} \)
2. \( T_i = 2 \times T \)
3. \( T_d = 0.5 \times T \)

E, por fim:

1. \( K_p \)
2. \( K_i = \frac{K_p}{T_i} \)
3. \( K_d = K_p \times T_d \)

## Métodos

### `ziegler_nichols_first_order(K, T) -> Tuple[float, float, float]`

Aplica o método de Ziegler-Nichols para sistemas de primeira ordem sem tempo morto.

**Parâmetros**:

- **K (float)**: inclinação da tangente no início da curva de resposta ao degrau.
- **T (float)**: tempo para a resposta atingir 63,2% do seu valor final.

**Retorno**:

- Ganhos sintonizados \(K_p, K_i, K_d\).

### `auto_tune(K, T) -> None`

Sintoniza automaticamente o controlador PID associado usando os parâmetros \(K\) e \(T\).

**Parâmetros**:

- **K (float)**: inclinação da tangente no início da curva de resposta ao degrau.
- **T (float)**: tempo para a resposta atingir 63,2% do seu valor final.

## Exemplo de Uso

```python
# Supondo que você tenha um controlador PID já definido
pid_controller = PID()

# Criando o sintonizador
tuner = PIDAutoTuner(pid_controller)

# Suponha que após analisar a resposta ao degrau do seu sistema você determinou:
K = 2.0  # Inclinação da tangente
T = 1.5  # Tempo para atingir 63,2% da resposta

# Realize a sintonia automática
tuner.auto_tune(K, T)

# Seu controlador PID agora está sintonizado com os novos ganhos.
```
