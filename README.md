# Hybrid Federated-Distributed Reinforcement Learning Framework for LunarLander

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A novel hybrid federated-distributed reinforcement learning framework that combines federated learning's global model integration with distributed reinforcement learning's parallel exploration capabilities for the LunarLander-v2 environment.

## 🎯 Abstract

This framework addresses the limitations of traditional federated reinforcement learning by implementing a bidirectional learning architecture where both local clients and the global server perform independent learning. Unlike conventional approaches that rely on simple parameter averaging, our system leverages experience sharing and trust-aware aggregation to achieve faster convergence and better generalization.

## ✨ Key Features

- **Dual Learner Architecture**: Simultaneous learning at both local and global levels
- **Experience Data Aggregation**: Direct sharing of transition data for global learning
- **Trust-aware Parameter Aggregation**: Cosine similarity-based weighted aggregation
- **Transition Diversity Filtering**: Euclidean distance-based filtering (threshold: 0.15)
- **Asynchronous Update Scheduling**: Independent update cycles for experience learning (0.5s) and parameter aggregation (30s)
- **Multi-threaded Concurrency Control**: Hierarchical lock system for data consistency
- **Efficient Compression**: PyTorch state_dict serialization with gzip compression (~50% reduction)

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Global Server (FastAPI)                  │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │          Dual Learning Components                    │    │
│  │  ┌─────────────────┐    ┌─────────────────────┐    │    │
│  │  │ Experience-based │    │ Parameter-based    │    │    │
│  │  │ Learning (0.5s)  │    │ Aggregation (30s)  │    │    │
│  │  └────────┬────────┘    └──────────┬─────────┘    │    │
│  │           │                         │               │    │
│  │  ┌────────▼──────────────────────────▼──────────┐  │    │
│  │  │     Global Experience Replay Buffer          │  │    │
│  │  │   (with Transition Diversity Filtering)      │  │    │
│  │  └───────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  API Endpoints:                                              │
│  • /upload-transition   (Experience data upload)             │
│  • /upload-weights     (Parameter upload with timestamp)     │
│  • /download-params    (Global model distribution)           │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP/REST
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼──────┐   ┌───────▼──────┐   ┌──────▼───────┐
│   Client 1   │   │   Client 2   │   │  Client N    │
│              │   │              │   │              │
│ Local DQN/   │   │ Local DQN/   │   │ Local DQN/   │
│ Double DQN   │   │ Double DQN   │   │ Double DQN   │
│              │   │              │   │              │
│ LunarLander  │   │ LunarLander  │   │ LunarLander  │
│ Environment  │   │ Environment  │   │ Environment  │
└──────────────┘   └──────────────┘   └──────────────┘
```

## 📊 Performance Results

### 3-Client Configuration
| Method                                              | reward           | episode           | step                  | flops/episode     | total_flops      | reward/flops     |
|-----------------------------------------------------|------------------|-------------------|------------------------|-------------------|------------------|------------------|
| Basic Local DQN                                     | 190.181          | 528               | 1.98e+05               | 1.93e+09          | 1.04e+12         | 1.83e-10         |
| Basic Local Double DQN                              | 191.926          | 498               | 1.83e+05               | 1.78e+09          | 8.86e+11         | 2.17e+10         |
| 3Fed DQN (CCN-FedAvg)                               | 126.86 ± 7.37    | 684.33 ± 12.62     | 3.38e+05 ± 6.58e+03     | 0.415             | 0.634            | 0.527            |
| 3Fed Double (CCN-FedAvg)                            | 77.51 ± 0.39     | 1.33e+03 ± 184.87  | 6.86e+05 ± 6.65e+04     | 0.379             | 0.367            | 0.610            |
| **3Fusion DQN (Fed trust & buffer filter)**         | **217.93 ± 3.21**| **406.67 ± 3.50**  | **1.74e+05 ± 7.42e+03** | **0.575**         | **0.541**        | **0.459**        |
| **3Fusion Double DQN (Fed trust & buffer filter)**  | **211.87 ± 2.02**| **390.33 ± 9.50**  | **1.82e+05 ± 8.57e+03** | **0.468**         | **0.473**        | **0.556**        |

### 5-Client Configuration

| Method                                         | reward           | episode          | step               | flops/episode     | total_flops       | reward/flops      |
|-----------------------------------------------|------------------|------------------|--------------------|-------------------|-------------------|-------------------|
| Basic Local DQN                               | 190.181          | 528              | 1.98e+05           | 1.93e+09          | 1.04e+12          | 1.83e-10          |
| Basic Local Double DQN                        | 191.926          | 498              | 1.83e+05           | 1.78e+09          | 8.86e+11          | 2.17e+10          |
| 5Fed DQN (CCN-FedAvg)                         | 151.19 ± 4.29    | 715.0 ± 76.102   | 343289.0 ± 24509.9 | 0.568 ± 0.488     | 0.59 ± 0.51       | 0.392 ± 0.451     |
| 5Fed Double (CCN-FedAvg)                      | 151.92 ± 22.4    | 629.8 ± 103.35   | 281740.6 ± 66470.8 | 0.60 ± 0.408      | 0.56 ± 0.423      | 0.291 ± 0.41      |
| **5Fusion DQN** *(Fed trust&buffer filter)*   | **215.15 ± 1.38**| **402.2 ± 3.633**| **180887.2 ± 5232.5**| **0.372 ± 0.374**| **0.477 ± 0.357** | **0.489 ± 0.369** |
| **5Fusion Double DQN** *(Fed trust&buffer filter)* | **214.40 ± 1.62**| **419.4 ± 12.116**| **182899.0 ± 4413.7**| **0.498 ± 0.416**| **0.584 ± 0.456** | **0.468 ± 0.435** |

### Performance comparison depending on the use of buffer filtering
| Method                                              | reward           | episode           | step                  | flops/episode     | total_flops       | reward/flops      |
|-----------------------------------------------------|------------------|-------------------|------------------------|-------------------|-------------------|-------------------|
| 3Fusion Double DQN (Fed trust & buffer filter (x))  | 211.86 ± 2.01    | 390.333 ± 9.504   | 182201.33 ± 8566.5     | 0.5               | 0.473 ± 0.502     | 0.555 ± 0.509     |
| 3Fusion Double DQN (Fed trust & buffer filter (o))  | 214.82 ± 2.54    | 415.0 ± 15.395    | 189389.6 ± 10403.5     | 0.452 ± 0.507     | 0.494 ± 0.5       | 0.515 ± 0.501     |
| 5Fusion Double DQN (Fed trust & buffer filter (x))  | 219.96 ± 0.54    | 372.8 ± 5.02      | 170791 ± 9145.61       | 0.341 ± 0.389     | 0.383 ± 0.373     | 0.59 ± 0.366      |
| 5Fusion Double DQN (Fed trust & buffer filter (o))  | 214.40 ± 1.62    | 419.4 ± 12.116    | 182899.0 ± 4413.773    | 0.498 ± 0.416     | 0.584 ± 0.456     | 0.468 ± 0.435     |



## 🔧 Implementation Details

### Global Server Components

1. **Concurrency Control**
   - `buffer_lock`: Controls access to global replay buffer
   - `weight_lock`: Ensures parameter storage integrity
   - `aggregation_lock`: Maintains atomicity of parameter aggregation
   - Adaptive timeout mechanism to prevent deadlocks

2. **Communication Protocol**
   - RESTful API using FastAPI framework
   - Base64 encoding for JSON compatibility
   - Gzip compression for network efficiency

3. **Trust-aware Aggregation Formula**
   ```
   sim_k = cosine_similarity(θ_global, θ_k)
   w_k = max(0, sim_k) / Σ max(0, sim_i)
   θ_global_new = α·θ_global + (1-α)·Σ(w_k·θ_k)
   ```

### Transition Diversity Filtering

```
For new transition T = (s, a, r, s', d):
1. Sample k states {s₁, s₂, ..., sₖ} from buffer
2. Calculate D_t = (1/k)·Σ||s - sᵢ||₂
3. Accept T only if D_t ≥ θ (θ = 0.15)
```

## 📋 Experimental Setup

| Parameter | Value |
|-----------|-------|
| Total Episodes | 2000 |
| Max Steps per Episode | 1000 |
| Convergence Threshold | 200 (average over 100 episodes) |
| Local Parameter Upload | Every 15 seconds |
| Global Aggregation | Every 40 seconds |
| Experience Buffer Upload | Every 30 seconds |
| Buffer Structure | deque |
| Learning Rate | 0.001 |
| Batch Size | 32 |

## 🚀 Getting Started

### Prerequisites

- Python 3.9
- PyTorch 
- FastAPI
- OpenAI Gym (with LunarLander-v2)
- NumPy
- gzip, pickle, base64

### Installation

1. Clone the repository:
```bash
git clone https://github.com/hufs-FL/LanderLunchV2_global.git
cd LanderLunchV2_global
```

2. Install dependencies:
```bash
pip install --no-cache-dir \
    numpy \
    matplotlib \
    pandas \
    requests \
    fastapi \
    uvicorn \
    pydantic \
    cloudpickle \
    pyglet \
    scipy \
    pillow \
    tensorboard
```

### Running the System(Training Mode)
1. **Start the Global Server:**
```bash
# For Basic DQN
python global.py --agent_mode SIMPLE --network_mode SIMPLE --batch_size 64 --update_interval 0.5 --buffer_size 1000000 --port 5050

# For Double DQN
python global.py --test_mode=TRAIN --agent_mode DOUBLE --network_mode SIMPLE --batch_size 64 --update_interval 0.5 --buffer_size 1000000 --port 5050

# For Dueling DQN
python global.py --test_mode=TRAIN --agent_mode SIMPLE --network_mode DUELING --batch_size 64 --update_interval 0.5 --buffer_size 1000000 --port 5050

# For Double Dueling DQN
python global.py --test_mode=TRAIN --agent_mode DOUBLE --network_mode DUELING --batch_size 64 --update_interval 0.5 --buffer_size 1000000 --port 5050
```

## 🔬 Research Contributions

1. **Hybrid Architecture**: First to combine experience sharing with parameter aggregation in federated RL
2. **Dual Learning**: Simultaneous learning at both local and global levels
3. **Trust-aware Aggregation**: Novel weighting scheme based on policy similarity
4. **Diversity Filtering**: Maintains experience variety in global buffer

## 📚 Citation

If you use this code in your research, please cite:


@article{jeon2025hybrid,
  title={A Hybrid Federated-Distributed Reinforcement Learning Framework 
         for Efficient Policy Generalization and Global Convergence},
  author={Jeon, Byunghwan},
  journal={Hankuk University of Foreign Studies},
  year={2025}
}

## 🏭 Applications

- **Smart Factories**: Real-time adaptive control with rapid user requirement adaptation
- **Autonomous Driving**: Distributed learning from multiple vehicles
- **Large-scale IoT**: Sensor-based systems with non-sensitive data
- **Robotics**: Multi-robot coordination and policy learning

## ⚠️ Limitations and Considerations

1. **Security Trade-off**: Unlike traditional FL, this framework shares transition data, which may have privacy implications
2. **Network Overhead**: Continuous experience sharing requires stable network connectivity
3. **Scalability**: Tested with up to 5 clients; larger scales require additional optimization

## 🔮 Future Work

-  Extension to high-dimensional continuous control environments
-  Integration with policy gradient methods (PPO, SAC)
-  Communication efficiency optimization
-  Enhanced security features for sensitive applications
-  Distributed global server architecture

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <i>Advancing the frontier of federated reinforcement learning through efficient hybrid architectures</i>
</p>
