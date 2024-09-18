# two_player_game_wrapper.py

class TwoPlayerGameWrapper:
    def __init__(
        self, 
        game_class, 
        agents,
        num_games=4, 
        **game_kwargs
    ):
        """
        Initialize the two-player game wrapper.

        Args:
            game_class: The class of the game to be played.
            agents: A dictionary mapping player IDs to agent instances.
            num_games (int): Number of rounds to play.
            **game_kwargs: Additional arguments to pass to the game constructor.
        """
        # build game
        self.game = game_class(**game_kwargs)
        self.agents = agents  # {player_id: agent_instance}
        self.num_games = num_games

        assert len(self.agents) == 2, \
            f"This is a two-player game wrapper. Please provide exactly two players. {len(self.agents)} players were provided"


    def swap_agents(self):
        """Swap the agents assigned to player IDs."""
        self.agents[0], self.agents[1] = self.agents[1], self.agents[0]

    def play_game(self):
        """
        Play the game using the agents.

        Returns:
            detailed_log: A dictionary containing the detailed logs of all games, including per-agent logs.
            high_level_logs: A list of dictionaries containing high-level summaries of each game.
        """
        # Initialize agent logs per agent_id
        agent_logs = {agent.agent_id: [] for agent in self.agents.values()}
        agent_scores = {agent.agent_id: 0 for agent in self.agents.values()}
        agent_scores['num_turns'] = []


        for game_id in range(self.num_games):
            # reset the game 
            player_prompts = self.game.reset() # Should return a dict {player_id: prompt}

            # Reset agents with their respective prompts
            self.agents[0].reset(player_prompts[0])
            self.agents[1].reset(player_prompts[1])

            done = False 
            state = ""

            episode_logs = {agent.agent_id: [] for agent in self.agents.values()}
            while not done:
                for player_id, agent in enumerate(self.agents.values()):
                    # get the valid actions
                    valid_actions = self.game.get_valid_actions(player_id=player_id)

                    # get action from the agent
                    action, full_state = agent.get_action(
                        state=state,
                        valid_actions=valid_actions
                    )

                    # execute the action in the game
                    new_state, reward, done, info = self.game.step(player_id=player_id, action=action)

                    # log it to the agent
                    agent_id = agent.agent_id 
                    episode_logs[agent_id].append({
                        "state": state,
                        "full_state": full_state,
                        "action": action,
                        "info": info
                    })

                    if done:
                        break 

                    # update state
                    state = new_state

            # Game finished, add reward to all agent logs and the agent scores
            for player_id, agent in enumerate(self.agents.values()):
                agent_id = agent.agent_id 
                for i in range(len(episode_logs[agent_id])):
                    episode_logs[agent_id][i]['reward'] = reward[player_id]
                # extend agent logs
                agent_logs[agent_id].extend(episode_logs[agent_id])
                # add agent scores
                agent_scores[agent_id] += reward[player_id]

            agent_scores["num_turns"].append(self.game.get_info())

            # swap players
            self.swap_agents()


        return agent_logs, agent_scores
