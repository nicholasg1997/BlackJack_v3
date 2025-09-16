from stable_baselines3.common.callbacks import BaseCallback

class BlackjackMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.balance_sum = 0
        self.wins = 0
        self.losses = 0
        self.pushes = 0
        self.busts = 0
        self.bet_sum = 0
        self.episode_count = 0
        self.step_count = 0

    def _on_step(self):
        infos = self.locals["infos"]
        for info in infos:
            if "balance" in info:
                self.balance_sum += info["balance"]
            if "bet_placed" in info:  # Bet phase
                self.bet_sum += info["bet_placed"]
                self.step_count += 1
            if info.get("win", False):
                self.wins += 1
                self.episode_count += 1
            elif info.get("loss", False):
                self.losses += 1
                self.episode_count += 1
            elif info.get("push", False):
                self.pushes += 1
                self.episode_count += 1
            if info.get("is_bust", False):
                self.busts += 1

        if self.step_count > 0 and self.n_calls % 100 == 0:
            avg_balance = self.balance_sum / max(self.step_count, 1)
            win_rate = self.wins / max(self.episode_count, 1)
            draw_rate = self.pushes / max(self.episode_count, 1)
            bust_rate = self.busts / max(self.episode_count, 1)
            avg_bet = self.bet_sum / max(self.step_count, 1)
            self.logger.record("blackjack/avg_balance", avg_balance)
            self.logger.record("blackjack/win_rate", win_rate)
            self.logger.record("blackjack/bust_rate", bust_rate)
            self.logger.record("blackjack/avg_bet", avg_bet)
            self.logger.record("blackjack/draw_rate", draw_rate)

        return True

class ReturnMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.total_winnings = 0
        self.total_wagered = 0
        self.num_episodes = 0
        self.num_rollouts = 0
        self.ev_per_hand = 0
        self.roi = 0

    def reset(self):
        self.total_winnings = 0
        self.total_wagered = 0
        self.num_episodes = 0
        self.ev_per_hand = 0
        self.roi = 0

    def _on_rollout_start(self) -> None:
        self.total_winnings = 0
        self.total_wagered = 0
        self.num_episodes = 0
        self.num_rollouts = 0

    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        rewards = self.locals["rewards"]

        for i, done in enumerate(dones):
            if done:
                self.total_winnings += infos[i]['winnings']
                if 'bet_placed' in infos[i]:
                    self.total_wagered += infos[i]['bet_placed']
                self.num_episodes += 1

        self.num_rollouts += 1
        return True

    def _on_rollout_end(self) -> None:
        if self.num_episodes > 0:
            self.ev_per_hand = self.total_winnings / self.num_episodes
            self.roi = self.total_winnings / self.total_wagered if self.total_wagered > 0 else 0
            self.logger.record("blackjack/ev_per_hand", self.ev_per_hand)
            self.logger.record("blackjack/roi", self.roi)
            if 'eval_env' in self.locals:
                self.locals['eval_ev'] = self.ev_per_hand


