import random
import numpy as np
import os
import threading
import json
import requests
import time
import uuid
import gzip
import base64
import pickle


import torch as T
from torch.utils.tensorboard import SummaryWriter

from collections import deque


from model import *
from constants import *
from utils import ReplayBuffer

class Agent():
    def __init__(self, input_dims, n_actions, seed, agent_mode=SIMPLE, network_mode=SIMPLE, test_mode=False, batch_size=64, n_epochs=5, 
                 update_every=5, lr=0.0005, fc1_dims=64, fc2_dims=64, gamma=0.99, epsilon=1.0, eps_end=0.01, eps_dec=0.995, 
                 max_mem_size=1_00_000, tau=1e-3,    
                 # 글로벌 전송 인자 추가 부 -----------------------------
                 # 글로벌 전송 관련 인자
                 upload_interval_sec=15.0,
                 download_interval_sec=40.0,
                #  trans_url="http://192.168.50.64:5050/upload-transition",
                #  weights_url="http://192.168.50.64:5050/upload-weights",
                #  download_url="http://192.168.50.64:5050/download-params",
                 trans_url="http://127.0.0.1:5050/upload-transition",
                 weights_url="http://127.0.0.1:5050/upload-weights",
                 download_url="http://127.0.0.1:5050/download-params"):
                 
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.seed = random.seed(seed)
        
        self.agent_mode=agent_mode
        self.network_mode=network_mode
        self.test_mode=test_mode
        
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.update_every = update_every
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.mem_size = max_mem_size
        self.tau = tau

        self.pending_download = None

        
        # For naming purpose
        agent_ = '{}-'.format(self.agent_mode) if self.agent_mode!=SIMPLE else ''
        network_ = '{}-'.format(self.network_mode) if self.network_mode!=SIMPLE else ''
        self.agent_name = f'{agent_}{network_}DQN'.strip()

        base_name = f"{self.agent_mode}-{self.network_mode}-DQN"
        self.client_id  = f"{base_name}-{uuid.uuid4()}"
        self.agent_name = base_name   
        
        if network_mode==DUELING:
            self.Q_eval = DuelingDeepQNetwork(input_dims=input_dims, n_actions=n_actions, seed=seed, lr=lr, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
            if not test_mode:
                self.Q_next = DuelingDeepQNetwork(input_dims=input_dims, n_actions=n_actions, seed=seed, lr=lr, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        else:
            self.Q_eval = DeepQNetwork(input_dims=input_dims, n_actions=n_actions, seed=seed, lr=lr, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
            if not test_mode:
                self.Q_next = DeepQNetwork(input_dims=input_dims, n_actions=n_actions, seed=seed, lr=lr, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        
        if not test_mode:
            
            self.tensorboard_step = 0
            self.tensorboard_writer = SummaryWriter(log_dir=f'logs/{self.agent_name}')
            
            self.update_cntr = 0
            self.memory = ReplayBuffer(max_mem_size, batch_size, n_actions, seed)


        # 글로벌 전송 관련 설정-----------------------------------------------------------------------
        # 업로드 설정
        self.upload_interval = upload_interval_sec
        self.upload_interval_sec = upload_interval_sec
        self.trans_url = trans_url
        self.weights_url = weights_url
        self.current_transitions = []
        self.last_upload_time = time.time()
        
        # 통신 성능 지표 저장 
        self.perf_metrics = {
            'rtt_times': [],
            'processing_times': [],
            'total_times': [],
            'data_sizes': [],
            'throughputs': [],
            'failures': 0,
            'sync_count': 0
        }

        # 다운로드 설정
        self.download_interval = download_interval_sec
        self.download_url = download_url
        self.last_download_time = time.time()
        
        # 주기적 업로드 스레드 시작
        threading.Thread(target=self._periodic_upload, daemon=True).start()
        threading.Thread(target=self._periodic_download, daemon=True).start()

        # 학습 카운터
        self.update_cntr = 0
    
    def save_model(self):
        # Create models folder
        if not os.path.isdir(f'models/{self.agent_name}'):
            os.makedirs(f'models/{self.agent_name}')
        T.save(self.Q_eval.state_dict(), f'./models/{self.agent_name}/{self.agent_name}_EVAL.pth')
        T.save(self.Q_next.state_dict(), f'./models/{self.agent_name}/{self.agent_name}_TARGET.pth')
    
    def load_model(self):
        if os.path.exists(f'./models/{self.agent_name}/{self.agent_name}_EVAL.pth'):
            self.Q_eval.load_state_dict(T.load(f'./models/{self.agent_name}/{self.agent_name}_EVAL.pth', map_location=T.device(self.Q_eval.device)))
        if not self.test_mode:
            if os.path.exists(f'./models/{self.agent_name}/{self.agent_name}_TARGET.pth'):
                self.Q_next.load_state_dict(T.load(f'./models/{self.agent_name}/{self.agent_name}_TARGET.pth', map_location=T.device(self.Q_eval.device)))
            
    def on_epsiode_end(self, reward_avg, reward_min, reward_max, n_steps, i_steps):
        if not self.test_mode:
            self.tensorboard_writer.add_scalar('Reward Avg.', reward_avg, self.tensorboard_step)
            self.tensorboard_writer.add_scalar('Reward Min.', reward_min, self.tensorboard_step)
            self.tensorboard_writer.add_scalar('Reward Max.', reward_max, self.tensorboard_step)
            self.tensorboard_writer.add_scalar('Total Steps', n_steps, self.tensorboard_step)
            self.tensorboard_writer.add_scalar('Steps per Episode', i_steps, self.tensorboard_step)
            self.tensorboard_writer.add_scalar('Epsilon', self.epsilon, self.tensorboard_step)

        # --- 글로벌 전송 주기 확인 ---
        ep = self.tensorboard_step + 1
        if ep % self.upload_interval_sec == 0 and self.current_transitions:
            print(f"[INFO] Sending {len(self.current_transitions)} transitions at episode {ep}")
            # filtered = self._filter_transitions(self.current_transitions)
            self._send_transitions_async(self.current_transitions)

        self.current_transitions = []

    #파라미터 전송
    def _periodic_upload(self):
        while True:
            time.sleep(1.0)
            now = time.time()
            if now - self.last_upload_time >= self.upload_interval:
                # transitions 업로드
                if self.current_transitions:
                    try:
                       requests.post(self.trans_url, json=self.current_transitions, timeout=5)
                    except Exception as e:
                        print(f"[ERROR] Transition upload failed: {e}")
                        
                # 파라미터 압축코드 
                def compress_weights(state_dict):
                    """state_dict를 압축하고 base64로 인코딩"""
                    pickled = pickle.dumps(state_dict)
                    compressed = gzip.compress(pickled, compresslevel=6)
                    encoded = base64.b64encode(compressed).decode('utf-8')
                    return encoded
                
                # weights 업로드
                try:
                    sd = self.Q_eval.state_dict()
                    sd_json = {k: v.detach().cpu().numpy().tolist() for k, v in sd.items()}
                    weights_base64 = compress_weights(sd_json)
                    payload = {"client_id": self.client_id,  "weights_base64": weights_base64}
                    requests.post(self.weights_url, json=payload, timeout=5)
                except Exception as e:
                    print(f"[ERROR] Weights upload failed: {e}")
                # 초기화
                self.current_transitions.clear()
                self.last_upload_time = now

    # #파라미터 업데이트
    # def _periodic_download(self):
    #     while True:
    #         time.sleep(1.0)
    #         if time.time() - self.last_download_time < self.download_interval:
    #             continue

    #         try:
    #             resp = requests.get(self.download_url, timeout=5)
    #             resp.raise_for_status()
    #             payload = resp.json()
    #             sd_json = payload['state_dict']

    #             if not sd_json:
    #                 print("[WARN] 다운로드된 state_dict 가 비어 있습니다.")
    #             else:
    #                 state_dict = self.Q_eval.state_dict()
    #                 with T.no_grad():

    #                     for k, v in sd_json.items():
    #                         if k in state_dict:
    #                             state_dict[k] = T.tensor(v, dtype=T.float32)

    #                     self.Q_eval.load_state_dict(state_dict)
    #                     self.Q_next.load_state_dict(state_dict)

    #                 self.Q_eval.train()
    #                 self.Q_next.train()

    #                 print(f"[SYNC] Downloaded & loaded global params at {time.ctime()}")
    #                 print("successful weight update")
    #         except Exception as e:
    #             print(f"[ERROR] Param download failed: {e}")

    #         self.last_download_time = time.time()

    def _periodic_download(self):
        while True:
            time.sleep(1.0)
            if time.time() - self.last_download_time < self.download_interval:
                continue

            try:
                # 지연 시간 측정 시작
                start_time = time.time()
                
                # 데이터 크기 측정을 위한 설정
                req_start = time.time()
                resp = requests.get(self.download_url, timeout=5)
                req_end = time.time()
                
                resp.raise_for_status()
                payload = resp.json()
                sd_json = payload['state_dict']  # 또는 'weights'로 수정
                
                # 데이터 크기 계산 (bytes)
                data_size = len(json.dumps(payload).encode('utf-8'))
                
                if not sd_json:
                    print("[WARN] 다운로드된 state_dict 가 비어 있습니다.")
                else:
                    # 처리 시간 측정
                    process_start = time.time()
                    state_dict = {
                        k: T.tensor(v, dtype=T.float32) if not isinstance(v, T.Tensor) else v
                        for k, v in sd_json.items()
                    }
                    self.Q_eval.load_state_dict(state_dict)
                    self.Q_next.load_state_dict(state_dict)
                    process_end = time.time()
                    
                    # 총 지연 시간
                    total_time = time.time() - start_time
                    
                    # 성능 측정값 저장
                    if hasattr(self, 'perf_metrics'):
                        self.perf_metrics['rtt_times'].append((req_end - req_start)*1000)
                        self.perf_metrics['processing_times'].append((process_end - process_start)*1000)
                        self.perf_metrics['total_times'].append(total_time*1000)
                        self.perf_metrics['data_sizes'].append(data_size/1024)
                        self.perf_metrics['throughputs'].append((data_size/1024)/(total_time))
                        self.perf_metrics['sync_count'] += 1
                    
                # 동기화 횟수 카운터 증가
                self.sync_count = getattr(self, 'sync_count', 0) + 1
                
            except Exception as e:
                # 실패 카운터 증가
                self.sync_failures = getattr(self, 'sync_failures', 0) + 1
                print(f"[ERROR] Param download failed: {e}")
                if self.perf_metrics['sync_count'] + self.perf_metrics['failures'] > 0:
                    failure_rate = (self.perf_metrics['failures'] / 
                                (self.perf_metrics['failures'] + self.perf_metrics['sync_count'])) * 100
                print(f"[PERF] Failure Rate: {(self.sync_failures/(self.sync_failures+self.sync_count))*100:.2f}%")

            self.last_download_time = time.time()


    # def step(self, state, action, reward, next_state, done):
    #     if not self.test_mode:
    #         self.memory.add(state, action, reward, next_state, done)
    #     self.current_transitions.append({
    #         "state": state.tolist() if hasattr(state, 'tolist') else state,
    #         "action": int(action),
    #         "reward": float(reward),
    #         "next_state": next_state.tolist() if hasattr(next_state, 'tolist') else next_state,
    #         "done": bool(done)
    #     })
    #     self.update_cntr    = (self.update_cntr    + 1) % self.update_every
    #     if self.update_cntr    == 0 and len(self.memory) > self.batch_size:
    #         experiences = self.memory.sample()
    #         self.learn(experiences)
    
    def step(self, state, action, reward, next_state, done):
        if not self.test_mode:
            self.memory.add(state, action, reward, next_state, done)

        self.current_transitions.append({
            "state": state.tolist() if hasattr(state, 'tolist') else state,
            "action": int(action),
            "reward": float(reward),
            "next_state": next_state.tolist() if hasattr(next_state, 'tolist') else next_state,
            "done": bool(done)
        })

        # 안전한 시점에서 글로벌 파라미터 적용
        if hasattr(self, 'pending_download') and self.pending_download:
            with T.no_grad():
                state_dict = self.Q_eval.state_dict()
                for k, v in self.pending_download.items():
                    if k in state_dict:
                        state_dict[k] = T.tensor(v, dtype=T.float32)
                self.Q_eval.load_state_dict(state_dict)
                self.Q_next.load_state_dict(state_dict)
            self.pending_download = None
            self.Q_eval.train()
            self.Q_next.train()
            print(f"[SYNC] 글로벌 파라미터를 안전하게 적용함 at {time.ctime()}")

        self.update_cntr = (self.update_cntr + 1) % self.update_every
        if self.update_cntr == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)


    def replace_target_network(self):
        if self.update_every != 0 and self.update_cntr % self.update_every == 0:
            # Soft Update
            for target_param, local_param in zip(self.Q_next.parameters(), self.Q_eval.parameters()):
                target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
    
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = observation[np.newaxis,:] # Add an axis to pass to model
            self.Q_eval.eval()
            with T.no_grad():
                Q = self.Q_eval.forward(T.from_numpy(state).to(self.Q_eval.device))
            self.Q_eval.train()
            action = T.argmax(Q).item()
        else:
            action = np.random.choice(np.arange(self.n_actions))

        return action
    
    def epsilon_decay(self):
        self.epsilon = max(self.epsilon*self.eps_dec, self.eps_end)
    
    # def learn(self, samples):
    #     T.autograd.set_detect_anomaly(True)

    #     states, actions, rewards, next_states, dones = samples

    #     if self.agent_mode == DOUBLE:
    #         # Double DQN Approach
    #         self.Q_eval.eval()
    #         with T.no_grad():
    #             # Q_Eval over next states to fetch max action arguement to pass to q_next
    #             q_pred = self.Q_eval.forward(next_states).to(self.Q_eval.device)
    #             max_actions = T.argmax(q_pred, dim=1).long().unsqueeze(1)
    #             # Q_Target over next states from actions will be taken based on q_pred's max_actions
    #             q_next = self.Q_next.forward(next_states).to(self.Q_eval.device)
    #         self.Q_eval.train()
    #         q_target = rewards + \
    #             self.gamma*q_next.gather(1, max_actions)*(1.0 - dones)
    #     else:
    #         # DQN Approach
    #         q_target_next = self.Q_next.forward(next_states).to(self.Q_eval.device).detach().max(dim=1)[0].unsqueeze(1)
    #         # q_target = rewards + (self.gamma* q_target_next * (1 - dones))
    #         # Use torch.mul() to avoid in-place operation issues
    #         gamma_term = T.mul(self.gamma, q_target_next)
    #         done_term = T.mul(gamma_term, (1 - dones))
    #         q_target = T.add(rewards, done_term)

    #     # Training
    #     for epoch in range(self.n_epochs):
    #         q_eval = self.Q_eval.forward(states).to(self.Q_eval.device).gather(1, actions)
    #         loss = self.Q_eval.loss(q_eval, q_target).to(self.Q_eval.device)
    #         self.Q_eval.optimizer.zero_grad()
    #         loss.backward()
    #         self.Q_eval.optimizer.step()

    #     # Replace Target Network
    #     self.replace_target_network()

    def learn(self, samples):
        T.autograd.set_detect_anomaly(True)

        # 안전하게 먼저 디바이스로 이동
        states, actions, rewards, next_states, dones = samples
        device = self.Q_eval.device

        # detach()를 사용하여 계산 그래프에서 분리된 새로운 텐서 생성
        states = states.clone().detach().to(device)
        actions = actions.clone().detach().to(device)
        rewards = rewards.clone().detach().to(device)
        next_states = next_states.clone().detach().to(device)
        dones = dones.clone().detach().to(device)

        if self.agent_mode == DOUBLE:
            # Double DQN
            self.Q_eval.eval()
            with T.no_grad():
                q_pred = self.Q_eval.forward(next_states)
                max_actions = T.argmax(q_pred, dim=1).long().unsqueeze(1)
                q_next = self.Q_next.forward(next_states)
            self.Q_eval.train()
            
            # 인플레이스 연산 대신 새 텐서 생성
            q_next_gathered = q_next.gather(1, max_actions)
            q_target = rewards + self.gamma * q_next_gathered * (1.0 - dones)
        else:
            # DQN
            with T.no_grad():
                q_target_next = self.Q_next.forward(next_states)
                q_target_max = q_target_next.max(dim=1)[0].unsqueeze(1)
                q_target = rewards + self.gamma * q_target_max * (1.0 - dones)

        # Training step
        for epoch in range(self.n_epochs):
            # 이전 결과를 완전히 분리시킴
            self.Q_eval.optimizer.zero_grad()
            
            # 새로운 순전파 계산
            q_eval = self.Q_eval.forward(states).gather(1, actions)
            loss = self.Q_eval.loss(q_eval, q_target)
            
            # 역전파 및 옵티마이저 업데이트
            loss.backward()
            self.Q_eval.optimizer.step()

        self.replace_target_network()

    # #  글로벌 전송 헬퍼 메서드
    # def _filter_transitions(self, transitions):
    #     #코사인 유사도 < threshold 인 전이만 선별
    #     selected = []
    #     for t in transitions:
    #         vec = np.array(t['state'], dtype=float)
    #         norm = np.linalg.norm(vec) + 1e-8
    #         if not selected:
    #             selected.append(t)
    #         else:
    #             sims = [
    #                 np.dot(vec, np.array(s['state'])) /
    #                 (norm * (np.linalg.norm(np.array(s['state']))+1e-8))
    #                 for s in selected
    #             ]
    #             if max(sims) < self.cosine_threshold:
    #                 selected.append(t)
    #     return selected

    def _send_transitions_async(self, transition_data):
        # 비동기 HTTP POST 전송 부분
        def _send():
            try:
                resp = requests.post(
                    url=self.trans_url,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(transition_data)
                )
                print(f"[SEND] Status: {resp.status_code}, Response: {resp.json()}")
            except Exception as e:
                print(f"[ERROR] Failed to send transitions: {e}")

        threading.Thread(target=_send, daemon=True).start() 
        
    # ----------- 통신 성능 최종 계산 ------------
    def calculate_performance_summary(self):
        """학습 종료 시 호출하여 통신 성능 요약 계산"""
        if not hasattr(self, 'perf_metrics') or not self.perf_metrics.get('sync_count'):
            print("[WARN] 성능 측정 데이터가 없습니다.")
            return {}
        
        # 측정값이 있는 경우만 평균 계산
        summary = {}
        
        if self.perf_metrics['rtt_times']:
            summary['avg_rtt_ms'] = sum(self.perf_metrics['rtt_times']) / len(self.perf_metrics['rtt_times'])
        
        if self.perf_metrics['processing_times']:
            summary['avg_processing_ms'] = sum(self.perf_metrics['processing_times']) / len(self.perf_metrics['processing_times'])
        
        if self.perf_metrics['total_times']:
            summary['avg_total_time_ms'] = sum(self.perf_metrics['total_times']) / len(self.perf_metrics['total_times'])
        
        if self.perf_metrics['throughputs']:
            summary['avg_throughput_kbps'] = sum(self.perf_metrics['throughputs']) / len(self.perf_metrics['throughputs'])
        
        # 동기화 횟수와 실패율
        summary['total_syncs'] = self.perf_metrics['sync_count']
        total_attempts = self.perf_metrics['sync_count'] + self.perf_metrics['failures']
        
        if total_attempts > 0:
            summary['failure_rate_pct'] = (self.perf_metrics['failures'] / total_attempts) * 100
        else:
            summary['failure_rate_pct'] = 0
        
        # 학습 시간이 있는 경우 동기화 빈도 계산
        if hasattr(self, 'train_time') and self.train_time > 0:
            summary['sync_frequency_per_min'] = self.perf_metrics['sync_count'] / (self.train_time / 60)
        
        # 성능 점수 계산 (낮은 지연시간, 높은 처리량, 낮은 실패율이 좋은 점수)
        if 'avg_total_time_ms' in summary and 'avg_throughput_kbps' in summary and 'failure_rate_pct' in summary:
            score = (
                (1000 / max(summary['avg_total_time_ms'], 10)) * 0.4 +  # 지연시간 역수 (작을수록 좋음)
                (summary['avg_throughput_kbps'] / 500) * 0.4 +          # 처리량 (클수록 좋음)
                ((100 - summary['failure_rate_pct']) / 100) * 0.2        # 성공률 (높을수록 좋음)
            ) * 100  # 0-100 점수로 정규화
            
            summary['performance_score'] = min(max(score, 0), 100)  # 0-100 사이로 제한
        
        print("\n===== 통신 성능 요약 =====")
        
        for metric, value in summary.items():
            if metric.endswith(('ms', 'pct', 'kbps')):
                print(f"{metric}: {value:.2f}")
            else:
                print(f"{metric}: {value}")
        
        if 'performance_score' in summary:
            print(f"종합 성능 점수: {summary['performance_score']:.2f}/100")
        
        return summary